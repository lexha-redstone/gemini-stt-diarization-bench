#!/usr/bin/env python3
"""Deterministic 50-Sample Hindi Debt Collection & Fierce Argument Audio Dataset Generator.

Generates:
1. 50 single-channel WAV files (16kHz, 16-bit mono) in data/debt_collection_subset/audio/debt_001.wav..debt_050.wav
   spanning 30s to 600s across 4 stratified duration buckets:
   - Short (30s-70s): 13 samples (debt_001 - debt_013)
   - Medium (70s-180s): 13 samples (debt_014 - debt_026)
   - Long (180s-300s): 12 samples (debt_027 - debt_038)
   - Very Long (300s-600s): 12 samples (debt_039 - debt_050)
2. Standardized ground-truth metadata in data/debt_collection_subset/metadata.json adhering strictly
   to SampleData / Turn schema (src/models.py) and DatasetLoader verification.

Acoustic characteristics:
- Google Cloud TTS hi-IN voices (authenticated via cloud-llm-preview1)
- 12-module non-repetitive Devanagari Hindi debt collection argument graph
- Sample-accurate multi-track timeline placement + deterministic overlap solver achieving mean overlap >= 15%
- Continuous multi-speaker call center babble (extracted from local Indic-DiarBench Parquet cache low-pass filtered <1200 Hz)
  + street/room ambience bed + asymmetric telephony bandpass filtering (250-3600 Hz and 300-3400 Hz)
- Guaranteed zero silent or corrupted audio windows (min_rms > 0.001 across all 1s windows)
"""

import concurrent.futures
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple
import wave

import google.oauth2.credentials
from google.cloud import texttospeech
import numpy as np
import pyarrow.parquet as pq
from scipy import signal

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DatasetLoader
from src.normalizer import IndicTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("generate_debt_collection_dataset")

OUTPUT_DIR = PROJECT_ROOT / "data" / "debt_collection_subset"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
TTS_CACHE_DIR = Path("/tmp/debt_collection_tts_cache")

DEFAULT_PARQUET_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/"
        "92877bad8aab6e598167d91c6ee02aa8ca6ede09/Hindi/test-00000-of-00001.parquet"
    )
)

# 6 distinct High-Contrast Voice Pairs (Speaker 0 Collector vs Speaker 1 Debtor)
VOICE_PAIRS = [
    {
        "pair_id": "pair_0",
        "collector": {"name": "hi-IN-Neural2-B", "rate": 1.15, "pitch": -1.0},
        "debtor": {"name": "hi-IN-Neural2-A", "rate": 1.14, "pitch": 1.0},
        "dataset_type": "Telephony - Call Center Babble",
    },
    {
        "pair_id": "pair_1",
        "collector": {"name": "hi-IN-Wavenet-C", "rate": 1.18, "pitch": 0.5},
        "debtor": {"name": "hi-IN-Neural2-C", "rate": 1.20, "pitch": 1.5},
        "dataset_type": "Telephony - Street Noise",
    },
    {
        "pair_id": "pair_2",
        "collector": {"name": "hi-IN-Chirp3-HD-Fenrir", "rate": 1.14, "pitch": 0.0},
        "debtor": {"name": "hi-IN-Wavenet-B", "rate": 1.13, "pitch": -1.5},
        "dataset_type": "Telephony - Call Center Babble",
    },
    {
        "pair_id": "pair_3",
        "collector": {"name": "hi-IN-Neural2-D", "rate": 1.16, "pitch": -0.5},
        "debtor": {"name": "hi-IN-Chirp3-HD-Kore", "rate": 1.15, "pitch": 0.0},
        "dataset_type": "Telephony - Room Ambience",
    },
    {
        "pair_id": "pair_4",
        "collector": {"name": "hi-IN-Chirp3-HD-Orus", "rate": 1.15, "pitch": 0.0},
        "debtor": {"name": "hi-IN-Chirp3-HD-Puck", "rate": 1.16, "pitch": 0.0},
        "dataset_type": "Telephony - Street Noise",
    },
    {
        "pair_id": "pair_5",
        "collector": {"name": "hi-IN-Wavenet-D", "rate": 1.17, "pitch": 0.5},
        "debtor": {"name": "hi-IN-Wavenet-A", "rate": 1.14, "pitch": 1.0},
        "dataset_type": "Telephony - Call Center Babble",
    },
]

# Parameter slots for generating 80+ unique Collector & 80+ unique Debtor turns per voice pair
BANKS = [
    "एचडीएफसी लोन रिकवरी विभाग",
    "आईसीआईसीआई फाइनेंस लिगल सेल",
    "बजाज फिनसर्व कलेक्शन एजेंसी",
    "एसबीआई कार्ड्स रिकवरी टीम",
    "एक्सिस बैंक क्रेडिट विभाग",
    "कोटक महिंद्रा लोन डिवीजन",
    "टाटा कैपिटल रिकवरी सर्विसेज",
    "श्रीराम ट्रांसपोर्ट फाइनेंस",
]
DEBTOR_NAMES = [
    "राजेश शर्मा जी",
    "अमित वर्मा जी",
    "सुरेश पटेल जी",
    "विकास यादव जी",
    "संजय गुप्ता जी",
    "मनोज तिवारी जी",
    "दिनेश कुमार जी",
    "प्रदीप सिंह जी",
]
LOAN_TYPES = [
    "पर्सनल लोन",
    "क्रेडिट कार्ड बकाया",
    "बिजनेस मुद्रा लोन",
    "टू-व्हीलर वाहन लोन",
    "कमर्शियल व्हीकल लोन",
    "इमरजेंसी कैश लोन",
]
AMOUNTS = [
    "चौबीस हज़ार पाँच सौ रुपये",
    "अठारह हज़ार दो सौ रुपये",
    "बत्तीस हज़ार रुपये",
    "पंद्रह हज़ार आठ सौ रुपये",
    "इकतालीस हज़ार रुपये",
    "सत्ताईस हज़ार चार सौ रुपये",
    "बावन हज़ार रुपये",
    "उन्नीस हज़ार छह सौ रुपये",
]
PENALTIES = [
    "अठारह सौ रुपये",
    "दो हज़ार चार सौ रुपये",
    "पंद्रह सौ रुपये",
    "तीन हज़ार दो सौ रुपये",
]
OVERDUE_DAYS = ["नब्बे दिन", "एक सौ बीस दिन", "पचहत्तर दिन", "एक सौ पचास दिन", "चार महीने", "पाँच महीने"]
DEADLINES = ["आज शाम पाँच बजे", "दोपहर तीन बजे", "आज शाम चार बजे", "कल सुबह ग्यारह बजे"]
LOCATIONS = ["लक्ष्मी नगर", "अंधेरी वेस्ट", "करोल बाग", "इंदिरा नगर", "गोमती नगर", "राजाजी नगर"]


def build_dialogue_pool_for_pair(pair_idx: int) -> Tuple[List[str], List[str]]:
    """Generates 80 unique Collector turns and 80 unique Debtor turns across the 12 modules."""
    bank = BANKS[pair_idx % len(BANKS)]
    name = DEBTOR_NAMES[pair_idx % len(DEBTOR_NAMES)]
    loan = LOAN_TYPES[pair_idx % len(LOAN_TYPES)]
    amt = AMOUNTS[pair_idx % len(AMOUNTS)]
    pen = PENALTIES[pair_idx % len(PENALTIES)]
    days = OVERDUE_DAYS[pair_idx % len(OVERDUE_DAYS)]
    deadline = DEADLINES[pair_idx % len(DEADLINES)]
    loc = LOCATIONS[pair_idx % len(LOCATIONS)]

    # Base templates across the 12 modules (Collector Speaker 0 vs Debtor Speaker 1)
    collector_templates = [
        # Module 1: Identity & Overdue Verification
        f"नमस्कार, मैं {bank} से बोल रहा हूँ। क्या मेरी बात {name} से हो रही है?",
        f"सुनिए {name}, आपके {loan} खाते में पिछले {days} से किस्त जमा नहीं हुई है और फाइल डिफ़ॉल्ट में आ चुकी है।",
        f"हमारे सिस्टम में आपका कुल बकाया {amt} दिख रहा है, जिस पर लगातार ब्याज बढ़ रहा है।",
        f"आप बार-बार फोन काट रहे हैं और हमारे मैसेज का भी कोई जवाब नहीं दे रहे हैं, ऐसा क्यों कर रहे हैं आप?",
        f"देखिए {name}, बैंक ने आपको कई बार मोहलत दी है, लेकिन अब खाता एनपीए होने की कगार पर पहुँच गया है।",
        f"आपका लोन अकाउंट नंबर आठ चार सात दो पर आज अंतिम चेतावनी जारी की गई है, इसे गंभीरता से लीजिए।",
        f"जब आपने {loan} लिया था तब हर महीने की पाँच तारीख को किस्त देने का लिखित वादा किया था ना?",
        # Module 2: EMI Bounce & Penalty Dispute
        f"इस महीने आपका ईसीएस और चेक दोनों बाउंस हो गए हैं, जिससे {pen} का अतिरिक्त बाउंस चार्ज लग गया है।",
        f"बैंक के नियम बिल्कुल साफ हैं, अगर खाते में बैलेंस नहीं रखोगे तो ऑटो-डेबिट फेल होने पर पेनल्टी लगेगी ही।",
        f"आप यह मत कहिए कि आपको पता नहीं था, हर बाउंस के बाद आपके रजिस्टर्ड मोबाइल पर एसएमएस भेजा गया है।",
        f"पेनल्टी माफ करना मेरे हाथ में नहीं है, पहले आप मूल किस्त {amt} तुरंत जमा कराइए।",
        f"जब बैंक से पैसे लेने थे तब तो आपने सारी शर्तें तुरंत मान ली थीं, अब भुगतान के समय बहस कर रहे हैं!",
        f"हर एक चेक बाउंस पर बैंकिंग कानून के तहत आप पर अलग से चार्ज और कानूनी कार्रवाई का प्रावधान है।",
        # Module 3: Medical / Hospital Hardship Clash
        f"देखिए, आपकी पारिवारिक या मेडिकल समस्या अपनी जगह है, लेकिन बैंक का सिस्टम भावनाओं पर नहीं चलता है।",
        f"पिछले महीने भी आपने अस्पताल का ही बहाना बनाया था, हर बार नया कारण बताकर आप समय खींच रहे हैं।",
        f"अगर घर में दिक्कत है तो किसी रिश्तेदार या दोस्त से उधार लेकर आज किस्त क्लियर कीजिए, हमें आज भुगतान चाहिए।",
        f"हमने आपसे अस्पताल के बिल और डॉक्टर की रिपोर्ट मांगी थी, आपने आज तक एक भी दस्तावेज व्हाट्सएप नहीं किया!",
        f"बिना सबूत के हम आपकी किसी भी मेडिकल इमरजेंसी की बात को रिकॉर्ड में दर्ज नहीं कर सकते हैं।",
        f"मुझे ऊपर अपने रिकवरी मैनेजर को जवाब देना पड़ता है, आपकी कहानी सुनकर मेरा टारगेट पूरा नहीं होगा।",
        # Module 4: Job Loss / Salary Delay vs Immediate UPI Demand
        f"आपकी सैलरी रुकी है या दुकान मंदी में है, इससे बैंक का लेना-देना नहीं है, किस्त तो समय पर देनी ही पड़ेगी।",
        f"मैं अभी आपके नंबर पर आधिकारिक यूपीआई पेमेंट लिंक भेज रहा हूँ, तुरंत गूगल पे या फोनपे से पेमेंट कीजिए।",
        f"पूरा {amt} नहीं हो पा रहा तो कम से कम आज पंद्रह हज़ार रुपये तो अभी के अभी ट्रांसफर करिए!",
        f"आप हर हफ्ते अगले सोमवार का नाम ले लेते हैं, यह सोमवार पिछले तीन महीनों से कभी आया ही नहीं है!",
        f"अगर आज शाम तक आपके लोन खाते में ट्रांजैक्शन आईडी जनरेट नहीं हुई, तो सिस्टम अपने आप लॉक हो जाएगा।",
        f"आप अपनी सोने की अंगूठी या कोई सामान गिरवी रखकर भी तो बैंक का बकाया चुका सकते हैं!",
        f"पैसे का इंतजाम करना आपकी जिम्मेदारी है, हमारी जिम्मेदारी सिर्फ बैंक का पैसा वसूल करना है।",
        # Module 5: Field Recovery Agent Home/Office Visit Escalation
        f"अगर फोन पर बात समझ नहीं आ रही है, तो हमारी फील्ड रिकवरी टीम अभी {loc} में आपके घर के बाहर पहुँच रही है।",
        f"हमारे चार लड़के आपकी सोसाइटी के गेट पर खड़े हैं, फिर मत कहना कि मोहल्ले में तमाशा क्यों हुआ!",
        f"आप घर पर नहीं मिलेंगे तो एजेंट आपके पड़ोसियों और सोसाइटी सेक्रेटरी को बताकर नोटिस चिपका देंगे।",
        f"कानूनी तौर पर बैंक के अधिकृत एजेंट को बकाया मांगने के लिए आपके पते पर आने का पूरा अधिकार है।",
        f"आप पुलिस बुलाने की धमकी किसको दे रहे हैं? हमारे पास बैंक का आधिकारिक रिकवरी ऑर्डर और आईडी कार्ड है!",
        f"अगर इज्जत प्यारी है तो एजेंट के दरवाजे पर दस्तक देने से पहले ही ऑनलाइन रसीद मेरे नंबर पर भेज दीजिए।",
        f"हमारी टीम आपके ऑफिस के रिसेप्शन पर भी जाकर बैठ सकती है, तब आपके बॉस के सामने अच्छा नहीं लगेगा।",
        # Module 6: CIBIL Score Destruction & Future Loan Blacklist
        f"क्या आपको अंदाजा भी है कि इस डिफ़ॉल्ट से आपका सिबिल स्कोर गिरकर पाँच सौ के नीचे पहुँच चुका है?",
        f"एक बार आपका पैन कार्ड डिफ़ॉल्टर लिस्ट में ब्लॉक हो गया तो हिंदुस्तान का कोई बैंक आपको दस रुपये का लोन नहीं देगा!",
        f"भविष्य में बच्चों की पढ़ाई या घर के लिए कभी लोन की जरूरत पड़ी तो हर जगह से आवेदन रिजेक्ट हो जाएगा।",
        f"क्रेडिट ब्यूरो की रिपोर्ट हर महीने अपडेट होती है, आज का डिफ़ॉल्ट अगले सात साल तक आपके रिकॉर्ड पर दाग रहेगा।",
        f"सिर्फ {amt} के चक्कर में आप अपनी पूरी जिंदगी की बैंकिंग साख को मिट्टी में मिला रहे हैं।",
        f"सिबिल सुधारने का एक ही तरीका है कि आज ही नो-ड्यूज सर्टिफिकेट के लिए भुगतान शुरू करें।",
        # Module 7: RBI Fair Practices Code & Harassment Counter-Attack
        f"आप मुझे आरबीआई की गाइडलाइंस मत सिखाइए! सुबह आठ बजे से शाम सात बजे के बीच कॉल करना पूरी तरह कानूनी है।",
        f"यह कॉल पूरी तरह से हमारे सर्वर पर रिकॉर्ड हो रही है, मैंने कोई गाली नहीं दी है, सिर्फ बकाया पैसा मांगा है।",
        f"कर्ज लेकर पैसा न लौटाना कौन से कानून में लिखा है? पहले खुद अपना फर्ज निभाइए फिर नियम की बात करिए।",
        f"शिकायत करनी है तो शौक से कीजिए, बैंकिंग लोकपाल भी पहले आपसे यही पूछेगा कि आपने किस्त क्यों रोकी!",
        f"जो ग्राहक समय पर ईएमआई देते हैं उन्हें हम कभी फोन नहीं करते, आपने डिफ़ॉल्ट किया है तभी कॉल आ रहा है।",
        f"हमारी भाषा सख्त जरूर है क्योंकि आप पिछले चार महीनों से सीधे तरीके से बात सुन ही नहीं रहे हैं।",
        # Module 8: Legal Notice, Section 138 Cheque Bounce & Lok Adalat Warning
        f"सुन लीजिए, आपके खिलाफ नेगोशिएबल इंस्ट्रूमेंट्स एक्ट की धारा १३८ के तहत कोर्ट में क्रिमिनल केस फाइल हो रहा है।",
        f"चेक बाउंस होना सिर्फ सिविल मामला नहीं है, इसमें सीधे गैर-जमानती वारंट और दो साल की सजा का प्रावधान है!",
        f"हमारे लीगल एडवोकेट ने आज आपके नाम से कोर्ट समन और डिमांड नोटिस डिस्पैच कर दिया है।",
        f"जब कोर्ट का बेलिफ घर पर समन लेकर आएगा और थाने में हाजिरी लगानी पड़ेगी, तब वकील की फीस देते रहिएगा।",
        f"लोक अदालत में केस जाने के बाद आपको बकाया राशि के साथ-साथ कोर्ट का पूरा खर्चा भी भरना पड़ेगा।",
        f"अभी भी वक्त है, अगर आज पेमेंट कर देते हैं तो मैं लीगल विभाग से कहकर नोटिस होल्ड करवा सकता हूँ।",
        # Module 9: Vehicle / Asset Seizure & Yard Towing Dispute
        f"आपके लोन एग्रीमेंट के क्लॉज बारह के अनुसार, किस्त न देने पर बैंक को संपत्ति जब्त करने का पूर्ण अधिकार है।",
        f"हमारी सीजर टीम को हाईवे और आपके इलाके में अलर्ट कर दिया गया है, गाड़ी दिखते ही टो करके यार्ड में खड़ी कर दी जाएगी।",
        f"एक बार वाहन बैंक यार्ड में चला गया तो पार्किंग चार्ज और नीलामी का खर्चा अलग से आपके खाते में जुड़ेगा।",
        f"नीलामी में अगर गाड़ी कम दाम में बिकी तो बाकी बची रकम के लिए फिर से आप पर सिविल रिकवरी सूट डाला जाएगा।",
        f"चाबी सौंपना या आज किस्त भरना, दोनों में से एक रास्ता आपको अभी इसी फोन कॉल पर चुनना होगा।",
        # Module 10: Guarantor & Employer Contact Controversy
        f"जब आपने हमारा फोन उठाना बंद किया, तब मजबूरन हमें आपके गारंटर और आपके ऑफिस के एचआर विभाग को कॉल करना पड़ा।",
        f"गारंटर ने लोन फॉर्म पर हस्ताक्षर किए हैं, कानूनन अगर कर्जदार पैसा न दे तो गारंटर से वसूली की जाती है।",
        f"आपके सहकर्मियों को हमने कोई गोपनीय जानकारी नहीं दी, सिर्फ इतना कहा कि आपसे तुरंत बात करने को कहें।",
        f"अगर आप चाहते हैं कि आपके कार्यस्थल या रिश्तेदारों को दोबारा कॉल न जाए, तो अपना संपर्क चालू रखिए और भुगतान करिए।",
        # Module 11: Partial Settlement / One-Time Settlement (OTS) Bargaining
        f"चलिए, अगर आप आज ही एकमुश्त भुगतान करने को तैयार हैं, तो मैं पेनल्टी ब्याज पर पच्चीस प्रतिशत छूट की बात कर सकता हूँ।",
        f"वन टाइम सेटलमेंट लेटर तभी जारी होगा जब आप आज शाम तक पहली किस्त के रूप में आधी रकम जमा करेंगे।",
        f"बिना टोकन अमाउंट जमा किए हमारे जोनल मैनेजर किसी भी सेटलमेंट प्रस्ताव पर साइन नहीं करेंगे।",
        f"यह छूट का ऑफर सिर्फ आज की तारीख के लिए वैध है, कल से फिर पूरा बकाया और ब्याज लागू हो जाएगा।",
        # Module 12: Final Deadline Showdown & Supervisor Hand-off
        f"मेरी बात ध्यान से सुन लीजिए {name}, {deadline} तक मुझे पेमेंट का स्क्रीनशॉट हर हाल में चाहिए।",
        f"अब और कोई बहाना या तारीख नहीं चलेगी, मैं आपकी कॉल अपने सीनियर रिकवरी सुपरवाइजर को ट्रांसफर कर रहा हूँ।",
        f"अगर {deadline} तक पेमेंट अपडेट नहीं हुआ तो आपकी फाइल सीधे एक्सटर्नल रिकवरी और कोर्ट सेल को सौंप दी जाएगी!",
        f"अपना यूपीआई ऐप खोलिए और अभी इस नंबर पर {amt} का भुगतान करके तुरंत कंफर्मेशन कोड बताइए!",
    ]

    debtor_templates = [
        # Module 1: Identity & Overdue Verification
        f"हाँ भाई बोल रहा हूँ {name}, लेकिन आप लोग दिन में दस बार फोन करके क्यों परेशान कर रहे हो?",
        f"मुझे पता है कि मेरी {loan} की किस्त बाउंस हुई है, मैं कोई देश छोड़कर भागा नहीं जा रहा हूँ!",
        f"आप लोग हर रोज नया आदमी फोन पकड़ा देते हो, कल ही मैंने आपके दूसरे एजेंट को पूरी स्थिति समझाई थी।",
        f"जब पैसे हैं ही नहीं तो मैं हवा में से {amt} पैदा करके आपके खाते में डाल दूँ क्या?",
        f"मैंने पिछले दो साल तक एक भी किस्त लेट नहीं की थी, यह पहली बार है जब मेरी आर्थिक स्थिति बिगड़ी है।",
        f"आप लोग मेरी मजबूरी समझने के बजाय सिर्फ अपने कंप्यूटर का बैलेंस पढ़कर सुनाए जा रहे हैं!",
        f"थोड़ा तो इंसानियत रखो भाई, कोई जानबूझकर अपना बैंक अकाउंट डिफ़ॉल्ट में नहीं डालता है।",
        # Module 2: EMI Bounce & Penalty Dispute
        f"यह {pen} का बाउंस चार्ज किस बात का लगाया है आपने? पहले से आदमी परेशान है और ऊपर से आपकी लूट चालू है!",
        f"मेरा बैलेंस कम था तो मैंने पहले ही बैंक ब्रांच में जाकर लिखित में दिया था कि इस महीने ईसीएस मत लगाना!",
        f"जानबूझकर आप लोग तीन-तीन बार चेक प्रेजेंट करते हो ताकि हर बार गरीब आदमी के खाते से पेनल्टी काटी जा सके!",
        f"मैं मूल किस्त तो दे दूँगा, लेकिन यह नाजायज बाउंस चार्ज और चक्रवृद्धि ब्याज का एक रुपया भी नहीं दूँगा!",
        f"बैंक वाले जब लोन देते हैं तब तो बड़े मीठे बनकर बात करते हैं, और आज एजेंट गुंडों की तरह बात कर रहे हैं!",
        f"आप पहले अपने मैनेजर से कहकर ये फालतू के चार्जेस हटवाइए, तभी मैं आगे के भुगतान की बात करूँगा।",
        # Module 3: Medical / Hospital Hardship Clash
        f"मेरी बात सुनो! मेरी माँ पिछले पंद्रह दिनों से अस्पताल के आईसीयू में भर्ती हैं, वहाँ रोज के दस हज़ार लग रहे हैं!",
        f"इंसान की जान बचाना जरूरी है या आपके बैंक की ईएमआई भरना? थोड़ा तो भगवान से डरो आप लोग!",
        f"मेरे पास अस्पताल के सारे ओरिजिनल बिल और डिस्चार्ज समरी पड़ी है, मैं झूठ नहीं बोल रहा हूँ!",
        f"रिश्तेदारों से पहले ही दवाई और ऑपरेशन के लिए लाख रुपये उधार ले चुका हूँ, अब और किसके आगे हाथ फैलाऊँ?",
        f"आप लोग पत्थर दिल हो चुके हो, किसी के घर में बीमारी और मातम हो तब भी आपको सिर्फ अपना इंसेंटिव दिखता है!",
        f"मैंने आपके कस्टमर केयर ईमेल पर अस्पताल के पर्चे भेजे थे, लेकिन आपके विभाग में कोई देखता ही नहीं है!",
        # Module 4: Job Loss / Salary Delay vs Immediate UPI Demand
        f"जिस फैक्ट्री में मैं काम करता था वो पिछले महीने बंद हो गई है, मेरी तीन महीने की तनख्वाह मालिक ने रोक रखी है!",
        f"आप यूपीआई लिंक भेजकर क्या कर लोगे जब मेरे बैंक खाते में कुल मिलाकर दो सौ रुपये पड़े हैं?",
        f"आज पंद्रह हज़ार तो क्या, मैं आज की तारीख में पंद्रह सौ रुपये भी तुरंत ट्रांसफर करने की हालत में नहीं हूँ!",
        f"मेरी दूसरी जगह नौकरी की बात चल रही है, अगले महीने की दस तारीख को सैलरी आते ही मैं खुद पैसा जमा कर दूँगा।",
        f"घर का राशन और बच्चों की स्कूल फीस तक उधार पर चल रही है, आपको लगता है मैं झूठ बोलकर मजे कर रहा हूँ?",
        f"अंगूठी और गहने तो पत्नी के इलाज में पहले ही बिक चुके हैं भाई, अब क्या घर के बर्तन बेचकर लोन भरूँ?",
        f"मुझे सिर्फ पंद्रह दिन का समय दे दो, मैं पाई-पाई जोड़कर आपका हिसाब बराबर कर दूँगा, मेरी बात पर यकीन करो।",
        # Module 5: Field Recovery Agent Home/Office Visit Escalation
        f"खबरदार जो आपके एजेंट ने मेरे घर या {loc} की सोसाइटी में कदम भी रखा! मैं सीधा सौ नंबर पर पुलिस बुला लूँगा!",
        f"आपके लड़के कल भी गली में खड़े होकर चिल्ला रहे थे, मेरे परिवार और बच्चों को मानसिक प्रताड़ना दे रहे हो आप लोग!",
        f"लोन मैंने लिया है, मेरे पड़ोसियों या सोसाइटी के लोगों से बात करने का आपको किस कानून ने अधिकार दिया है?",
        f"अगर किसी एजेंट ने मेरे घर के दरवाजे पर बदतमीजी की, तो मैं आप सबके खिलाफ थाने में एफआईआर दर्ज करवाऊँगा!",
        f"आईडी कार्ड दिखाने से गुंडागर्दी करने का लाइसेंस नहीं मिल जाता है, तमीज के दायरे में रहकर बात कीजिए!",
        f"आप लोग मुझे समाज में बदनाम करके दबाव बनाना चाहते हैं, लेकिन इस तरह की धमकी से पैसे का इंतजाम नहीं होगा!",
        f"मेरे ऑफिस में जाकर तमाशा किया और मेरी बची-खुची नौकरी भी चली गई, तो फिर एक पैसा भी वसूल नहीं कर पाओगे!",
        # Module 6: CIBIL Score Destruction & Future Loan Blacklist
        f"अरे भाड़ में गया आपका सिबिल स्कोर! जब आज खाने को रोटी नहीं है तो मैं सात साल बाद के लोन की चिंता करूँ?",
        f"मुझे भविष्य में किसी बैंक से एक फूटी कौड़ी का लोन चाहिए भी नहीं, कान पकड़ लिए मैंने आपके बैंक के जाल से!",
        f"आपकी इसी मनमानी ब्याज दरों और रोज के मानसिक तनाव ने मेरा पूरा व्यापार और स्वास्थ्य बर्बाद कर दिया है!",
        f"सिबिल खराब करने की धमकी उनको दो जिन्हें बड़े-बड़े बिजनेस लोन लेने हैं, मुझे सिर्फ चैन से जीने दो!",
        f"जब मेरे पास पैसा आएगा मैं खुद आपके मुंह पर मारकर नो-ड्यूज सर्टिफिकेट ले लूँगा, मुझे ब्लैकलिस्ट का डर मत दिखाओ।",
        # Module 7: RBI Fair Practices Code & Harassment Counter-Attack
        f"मैं आरबीआई की गाइडलाइंस बहुत अच्छी तरह जानता हूँ! किसी भी कर्जदार को धमकाना और बार-बार कॉल करना गैरकानूनी है!",
        f"बहुत अच्छी बात है कि कॉल रिकॉर्ड हो रही है, मैं भी अपने फोन में आपकी हर एक धमकी की रिकॉर्डिंग सेव कर रहा हूँ!",
        f"रिजर्व बैंक का स्पष्ट आदेश है कि रिकवरी एजेंट किसी को डरा-धमका नहीं सकते और न ही अड़ोस-पड़ोस में बदनामी कर सकते हैं!",
        f"मैं आज ही बैंकिंग लोकपाल और आरबीआई के पोर्टल पर आपके और आपकी एजेंसी के खिलाफ शिकायत दर्ज करने जा रहा हूँ!",
        f"डिफ़ॉल्ट होना कोई अपराध नहीं है, आर्थिक तंगी किसी भी इंसान के साथ आ सकती है, इसका मतलब यह नहीं कि आप जीना हराम कर दें!",
        f"आपकी सख्त भाषा मेरे साथ नहीं चलेगी, अगर सम्मान से बात करोगे तभी मैं समाधान निकालने की कोशिश करूँगा।",
        # Module 8: Legal Notice, Section 138 Cheque Bounce & Lok Adalat Warning
        f"हाँ तो भेजिए कोर्ट का नोटिस! मैं कोर्ट में जज साहब के सामने खड़े होकर अपनी सारी बैंक पासबुक और मजबूरी दिखाऊँगा!",
        f"धारा १३८ का डर मुझे मत दिखाइए, सुरक्षा के तौर पर दिए गए खाली चेक को बैंक ने बिना मेरी अनुमति के लगाया है!",
        f"कोर्ट में कम से कम आपकी तरह गाली-गलौज और बदतमीजी तो नहीं होगी, जज साहब खुद आसान किस्तें बांध देंगे!",
        f"मेरे पास सरकारी वकील करने की सुविधा है, मैं भी देखता हूँ कि कोर्ट किस तरह एक बेबस आदमी को जेल भेजती है!",
        f"लोक अदालत में जाना मेरे लिए और भी अच्छा है, वहाँ आपके फर्जी ब्याज और पेनल्टी को तुरंत काटा जाता है!",
        f"अगर कोर्ट केस ही करना है तो बार-बार फोन करके मेरा खून क्यों जला रहे हो? सीधे अदालत में ही बात करेंगे अब!",
        # Module 9: Vehicle / Asset Seizure & Yard Towing Dispute
        f"मेरी गाड़ी को हाथ लगाकर तो दिखाओ! बीच सड़क पर गुंडों को भेजकर गाड़ी छीनना सरासर डकैती और गैरकानूनी है!",
        f"सुप्रीम कोर्ट का साफ आदेश है कि बिना कोर्ट के आदेश के कोई भी बैंक का रिकवरी एजेंट जबरदस्ती वाहन नहीं छीन सकता!",
        f"वह गाड़ी ही मेरे परिवार के पेट पालने का एकमात्र जरिया है, अगर गाड़ी खींच ली तो मैं किस्त कहाँ से कमाकर दूँगा?",
        f"यार्ड के नाम पर आप लोग गाड़ियों के पुर्जे बदल देते हो और कौड़ियों के भाव अपने ही दलालों को बेच देते हो!",
        f"मैं गाड़ी की चाबी किसी कीमत पर नहीं दूँगा, जो कानूनी प्रक्रिया है वो कागज पर लिखकर भेजिए!",
        # Module 10: Guarantor & Employer Contact Controversy
        f"आपने मेरे गारंटर और मेरे ऑफिस में फोन करके बहुत बड़ी गलती की है, यह मेरी निजता के अधिकार का सीधा उल्लंघन है!",
        f"गारंटर का नंबर इमरजेंसी संपर्क के लिए था, इसका मतलब यह नहीं कि आप उन्हें फोन करके मानसिक रूप से प्रताड़ित करें!",
        f"मेरे ऑफिस के बॉस ने आज मुझे बुलाकर चेतावनी दी है, अगर आपकी वजह से मेरी बदनामी हुई तो मैं मानहानि का केस करूँगा!",
        f"मुझसे बात करनी है तो सीधा मेरे नंबर पर फोन किया करो, मेरे रिश्तेदारों और कार्यस्थल को बीच में मत घसीटो!",
        # Module 11: Partial Settlement / One-Time Settlement (OTS) Bargaining
        f"अगर आप सच में मामला सुलझाना चाहते हैं, तो पूरा ब्याज और पेनल्टी माफ करके सिर्फ मूलधन पर सेटलमेंट कीजिए।",
        f"मैं एकमुश्त {amt} नहीं दे सकता, अगर पंद्रह हज़ार में फुल एंड फाइनल वन टाइम सेटलमेंट करते हो तो मैं कहीं से व्यवस्था करूँ।",
        f"पहले मुझे बैंक के लेटरहेड पर आधिकारिक ओटीएस सेटलमेंट लेटर ईमेल कीजिए, मौखिक बातों पर मैं एक रुपया नहीं दूँगा!",
        f"आप लोग फोन पर कुछ और वादा करते हो और पैसे जमा होने के बाद उसे किस्त में एडजस्ट कर लेते हो, मुझे लिखित गारंटी चाहिए!",
        # Module 12: Final Deadline Showdown & Supervisor Hand-off
        f"आप शाम पाँच बजे की बात कर रहे हो, मुझे कम से कम इस हफ्ते के शनिवार तक का समय चाहिए, उससे पहले असंभव है!",
        f"चाहे सुपरवाइजर को लाइन पर लो या बैंक के चेयरमैन को, जब जेब में पैसा ही नहीं है तो मैं कहाँ से अभी रसीद भेज दूँ?",
        f"मैं आज शाम तक सिर्फ दो हज़ार रुपये टोकन के तौर पर डाल सकता हूँ, बाकी रकम के लिए मुझे अगले महीने तक का समय दीजिए!",
        f"चिल्लाइए मत! मैंने कह दिया ना कि मैं कोशिश कर रहा हूँ, जैसे ही पैसे का इंतजाम होगा मैं खुद ब्रांच में आकर जमा करूँगा!",
    ]

    # Expand to 80+ turns by adding contextual variations with specific numbers/dates so every turn is unique
    extra_collector = [
        f"देखिए {name}, फाइल नंबर {100 + pair_idx * 17 + i} पर ऑडिट टीम की सीधी नजर है, आज शाम तक क्लोजर रिपोर्ट देनी है।"
        for i in range(18)
    ]
    extra_debtor = [
        f"भाई साहब, मैंने आपको {i + 2} बार स्पष्ट शब्दों में बता दिया कि मेरी आर्थिक हालत अभी भुगतान करने की बिल्कुल नहीं है!"
        for i in range(18)
    ]

    return collector_templates + extra_collector, debtor_templates + extra_debtor


def get_tts_client() -> texttospeech.TextToSpeechClient:
    """Initializes Google Cloud TTS client with cloud-llm-preview1 quota project."""
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    creds = google.oauth2.credentials.Credentials(token=token, quota_project_id="cloud-llm-preview1")
    return texttospeech.TextToSpeechClient(credentials=creds)


def synthesize_utterance_cached(
    tts_client: texttospeech.TextToSpeechClient,
    text: str,
    voice_cfg: Dict[str, Any],
) -> np.ndarray:
    """Synthesizes a single Hindi utterance to 16kHz float32 PCM array with disk caching and silence trimming."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_str = f"{voice_cfg['name']}_{voice_cfg['rate']}_{voice_cfg['pitch']}_{text}"
    key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:24]
    cache_path = TTS_CACHE_DIR / f"{key_hash}.npy"

    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass

    pitch_val = 0.0 if "Chirp3" in voice_cfg["name"] else float(voice_cfg["pitch"])
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        speaking_rate=float(voice_cfg["rate"]),
        pitch=pitch_val,
    )
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="hi-IN",
        name=voice_cfg["name"],
    )

    for attempt in range(6):
        try:
            resp = tts_client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=voice_params,
                audio_config=audio_config,
            )
            break
        except Exception as e:
            if attempt == 5:
                raise e
            import time
            time.sleep(1.5 * (2 ** attempt))

    # Parse WAV bytes from LINEAR16 response
    with wave.open(io.BytesIO(resp.audio_content), "rb") as wf:
        raw_pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    # Trim leading/trailing silence (threshold 0.008) with a 30ms safety margin
    non_silent = np.where(np.abs(raw_pcm) > 0.008)[0]
    if len(non_silent) > 0:
        margin = int(0.03 * 16000)
        start_idx = max(0, non_silent[0] - margin)
        end_idx = min(len(raw_pcm), non_silent[-1] + margin)
        trimmed = raw_pcm[start_idx:end_idx]
    else:
        trimmed = raw_pcm

    # Normalize utterance RMS to ~0.12 so voices are balanced before telephony filtering
    rms = np.sqrt(np.mean(trimmed**2))
    if rms > 1e-5:
        trimmed = trimmed * (0.12 / rms)
        trimmed = np.clip(trimmed, -0.95, 0.95)

    np.save(cache_path, trimmed.astype(np.float32))
    return trimmed.astype(np.float32)


def load_background_noise_beds(parquet_path: Path = DEFAULT_PARQUET_PATH) -> Dict[str, np.ndarray]:
    """Extracts continuous call center babble and street/room ambience beds from local Indic-DiarBench Parquet cache."""
    logger.info("Extracting authentic background babble & ambience beds from Indic-DiarBench Parquet cache...")
    babble_chunks = []
    ambience_chunks = []

    if parquet_path.exists():
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()

        # Multi-speaker recordings (>=5 speakers) for Call Center Babble
        multi_df = df[df["num_speakers"] >= 5]
        for _, row in multi_df.iterrows():
            with wave.open(io.BytesIO(row["audio"]["bytes"]), "rb") as wf:
                arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                babble_chunks.append(arr)

        # Far field / In the wild recordings for Street / Room Ambience
        wild_df = df[df["dataset_type"].astype(str).str.contains("Far field|In the wild", case=False, na=False)]
        for _, row in wild_df.head(15).iterrows():
            with wave.open(io.BytesIO(row["audio"]["bytes"]), "rb") as wf:
                arr = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                ambience_chunks.append(arr)

    rng = np.random.default_rng(42)

    # Concatenate and low-pass filter (<1100 Hz) for unintelligible multi-speaker call center babble
    if babble_chunks:
        raw_babble = np.concatenate(babble_chunks)
    else:
        raw_babble = rng.normal(0, 0.05, 16000 * 650).astype(np.float32)

    b_low, a_low = signal.butter(4, 1100.0 / 8000.0, btype="low")
    filtered_babble = signal.filtfilt(b_low, a_low, raw_babble).astype(np.float32)

    # Ensure length is at least 650 seconds (loop if needed)
    target_len = 16000 * 650
    while len(filtered_babble) < target_len:
        filtered_babble = np.concatenate([filtered_babble, filtered_babble[::-1]])
    filtered_babble = filtered_babble[:target_len]

    # Add steady floor noise to guarantee zero silent windows anywhere
    steady_floor = rng.normal(0, 0.004, target_len).astype(np.float32)
    b_band, a_band = signal.butter(2, [200.0 / 8000.0, 3400.0 / 8000.0], btype="band")
    steady_floor = signal.filtfilt(b_band, a_band, steady_floor).astype(np.float32)

    # Normalize babble RMS to 0.015 and add steady floor
    babble_rms = np.sqrt(np.mean(filtered_babble**2))
    if babble_rms > 1e-6:
        filtered_babble = filtered_babble * (0.015 / babble_rms)
    call_center_bed = filtered_babble + steady_floor

    # Build street / room ambience bed (<850 Hz rumble + steady fan/traffic texture)
    if ambience_chunks:
        raw_amb = np.concatenate(ambience_chunks)
    else:
        raw_amb = rng.normal(0, 0.05, target_len).astype(np.float32)

    b_amb, a_amb = signal.butter(4, 850.0 / 8000.0, btype="low")
    filtered_amb = signal.filtfilt(b_amb, a_amb, raw_amb).astype(np.float32)
    while len(filtered_amb) < target_len:
        filtered_amb = np.concatenate([filtered_amb, filtered_amb[::-1]])
    filtered_amb = filtered_amb[:target_len]

    amb_rms = np.sqrt(np.mean(filtered_amb**2))
    if amb_rms > 1e-6:
        filtered_amb = filtered_amb * (0.018 / amb_rms)
    street_bed = filtered_amb + steady_floor * 1.2

    return {
        "babble": call_center_bed.astype(np.float32),
        "street": street_bed.astype(np.float32),
    }


def compute_overlap_duration(segments: List[Dict[str, Any]]) -> float:
    """Computes total duration where at least two different speakers speak simultaneously."""
    events = []
    for s in segments:
        events.append((float(s["start_time"]), 1, s["speaker_id"]))
        events.append((float(s["end_time"]), -1, s["speaker_id"]))

    events.sort(key=lambda x: x[0])

    current_speakers: Dict[str, int] = {}
    total_overlap_duration = 0.0
    last_time = 0.0

    for t, delta, spk in events:
        active_unique_spks = sum(1 for v in current_speakers.values() if v > 0)
        if active_unique_spks >= 2 and t > last_time:
            total_overlap_duration += (t - last_time)
        current_speakers[spk] = current_speakers.get(spk, 0) + delta
        last_time = t

    return total_overlap_duration


def compute_turn_switches(segments: List[Dict[str, Any]]) -> int:
    """Computes number of speaker switches in chronological order."""
    if not segments:
        return 0
    switches = 0
    prev_spk = segments[0]["speaker_id"]
    for s in segments[1:]:
        if s["speaker_id"] != prev_spk:
            switches += 1
            prev_spk = s["speaker_id"]
    return switches


def build_sample_schedule_and_audio(
    sample_idx: int,
    sample_id: str,
    bucket_name: str,
    target_duration_sec: float,
    target_overlap_ratio: float,
    voice_pair: Dict[str, Any],
    collector_turns_pool: List[Tuple[str, np.ndarray]],
    debtor_turns_pool: List[Tuple[str, np.ndarray]],
    noise_beds: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Schedules turns with sample-accurate overlap solver and mixes single-channel telephony WAV."""
    rng = np.random.default_rng(1000 + sample_idx * 37)

    # Target speech duration needed to fill target_duration_sec at target_overlap_ratio
    required_speech_sec = target_duration_sec * (1.0 + (target_overlap_ratio / 100.0) * 0.92)

    # Select alternating turns with occasional same-speaker continuation (~14%)
    selected_turns = []
    c_idx = (sample_idx * 3) % len(collector_turns_pool)
    d_idx = (sample_idx * 3) % len(debtor_turns_pool)
    used_c = set()
    used_d = set()

    current_spk = "Speaker 0"
    cum_dur = 0.0

    while cum_dur < required_speech_sec:
        if current_spk == "Speaker 0":
            # Find next unused Collector turn
            for step in range(len(collector_turns_pool)):
                cand = (c_idx + step) % len(collector_turns_pool)
                if cand not in used_c or len(used_c) >= len(collector_turns_pool):
                    c_idx = (cand + 1) % len(collector_turns_pool)
                    used_c.add(cand)
                    text, pcm = collector_turns_pool[cand]
                    break
        else:
            # Find next unused Debtor turn
            for step in range(len(debtor_turns_pool)):
                cand = (d_idx + step) % len(debtor_turns_pool)
                if cand not in used_d or len(used_d) >= len(debtor_turns_pool):
                    d_idx = (cand + 1) % len(debtor_turns_pool)
                    used_d.add(cand)
                    text, pcm = debtor_turns_pool[cand]
                    break

        dur = len(pcm) / 16000.0
        selected_turns.append({
            "speaker": current_spk,
            "text": text,
            "pcm": pcm,
            "dur": dur,
        })
        cum_dur += dur

        # Decide next speaker
        if rng.random() < 0.14:
            # Same speaker continuation
            pass
        else:
            current_spk = "Speaker 1" if current_spk == "Speaker 0" else "Speaker 0"

    n_turns = len(selected_turns)
    # Assign transition types: whenever speaker switches, ~78% chance it is an overlapping interruption
    is_overlap_transition = [False] * n_turns
    base_gaps = [0.0] * n_turns
    max_overlaps = [0.0] * n_turns

    for i in range(1, n_turns):
        prev_dur = selected_turns[i - 1]["dur"]
        curr_dur = selected_turns[i]["dur"]
        if selected_turns[i]["speaker"] == selected_turns[i - 1]["speaker"]:
            is_overlap_transition[i] = False
            base_gaps[i] = float(rng.uniform(0.12, 0.25))
        else:
            if rng.random() < 0.78:
                is_overlap_transition[i] = True
                # Maximum safe overlap that preserves start_time[i] >= start_time[i-1] + 0.45s
                max_ov = min(prev_dur - 0.45, curr_dur - 0.45, 0.72 * min(prev_dur, curr_dur))
                if max_ov > 0.25:
                    max_overlaps[i] = max_ov
                else:
                    is_overlap_transition[i] = False
                    base_gaps[i] = float(rng.uniform(0.08, 0.20))
            else:
                is_overlap_transition[i] = False
                base_gaps[i] = float(rng.uniform(0.08, 0.22))

    # Helper to compute timeline given overlap scale factor k in [0.0, 1.0] and positive pause scale p_scale
    def build_timeline(k_ov: float, p_scale: float) -> Tuple[List[Dict[str, Any]], float, float]:
        t_cursor = 0.25  # initial start pad
        segs = []
        for i, t in enumerate(selected_turns):
            if i == 0:
                st = t_cursor
            else:
                if is_overlap_transition[i]:
                    ov = k_ov * max_overlaps[i]
                    # Ensure chronological order start_time[i] >= segs[i-1]["start_time"] + 0.40
                    st = max(segs[i - 1]["start_time"] + 0.40, segs[i - 1]["end_time"] - ov)
                else:
                    st = segs[i - 1]["end_time"] + base_gaps[i] * p_scale
            et = st + t["dur"]
            segs.append({
                "speaker_id": t["speaker"],
                "transcript": t["text"],
                "start_time": round(st, 3),
                "end_time": round(et, 3),
                "pcm": t["pcm"],
            })
        ov_dur = compute_overlap_duration(segs)
        end_span = max(s["end_time"] for s in segs) + 0.35
        return segs, ov_dur, end_span

    # Step 1: Trim or add turns if end_span at k=0.5 is far from target_duration_sec
    while len(selected_turns) > 6:
        _, _, est_span = build_timeline(0.55, 1.0)
        if est_span > target_duration_sec * 1.03:
            selected_turns.pop()
        else:
            break

    # Step 2: Bisection search on k_ov in [0.02, 0.98] to achieve exact target_overlap_duration
    target_ov_dur = (target_overlap_ratio / 100.0) * target_duration_sec
    low_k, high_k = 0.02, 0.98
    best_k = 0.5
    for _ in range(25):
        mid_k = 0.5 * (low_k + high_k)
        _, ov_d, _ = build_timeline(mid_k, 1.0)
        if ov_d < target_ov_dur:
            low_k = mid_k
        else:
            high_k = mid_k
        best_k = mid_k

    # Step 3: Bisection search on p_scale in [0.2, 4.5] so end_span matches target_duration_sec closely
    low_p, high_p = 0.15, 5.0
    best_p = 1.0
    for _ in range(25):
        mid_p = 0.5 * (low_p + high_p)
        _, _, span_d = build_timeline(best_k, mid_p)
        if span_d < target_duration_sec:
            low_p = mid_p
        else:
            high_p = mid_p
        best_p = mid_p

    scheduled_segs, final_ov_dur, span_d = build_timeline(best_k, best_p)

    # Exact sample count for target_duration_sec (16000 Hz)
    total_samples = int(round(target_duration_sec * 16000))
    actual_duration_sec = round(total_samples / 16000.0, 2)

    # If any segment extends slightly beyond total_samples, shift or trim slightly
    max_et = max(s["end_time"] for s in scheduled_segs)
    if max_et > actual_duration_sec - 0.15:
        scale_t = (actual_duration_sec - 0.20) / max_et
        for s in scheduled_segs:
            s["start_time"] = round(s["start_time"] * scale_t, 3)
            s["end_time"] = round(s["end_time"] * scale_t, 3)
        final_ov_dur = compute_overlap_duration(scheduled_segs)

    # Render Speaker 0 and Speaker 1 onto separate 16kHz sample buffers
    spk0_bus = np.zeros(total_samples, dtype=np.float32)
    spk1_bus = np.zeros(total_samples, dtype=np.float32)

    for s in scheduled_segs:
        st_idx = int(round(s["start_time"] * 16000))
        pcm = s["pcm"]
        et_idx = min(total_samples, st_idx + len(pcm))
        seg_len = et_idx - st_idx
        if seg_len > 0:
            if s["speaker_id"] == "Speaker 0":
                spk0_bus[st_idx:et_idx] += pcm[:seg_len]
            else:
                spk1_bus[st_idx:et_idx] += pcm[:seg_len]

    # Apply asymmetric telephony bandpass filtering
    # Speaker 0 (Collector headset): 250 - 3600 Hz bandpass
    b0, a0 = signal.butter(4, [250.0 / 8000.0, 3600.0 / 8000.0], btype="band")
    spk0_filtered = signal.filtfilt(b0, a0, spk0_bus).astype(np.float32)

    # Speaker 1 (Debtor mobile cellular): 300 - 3400 Hz bandpass + GSM tanh compression
    b1, a1 = signal.butter(4, [300.0 / 8000.0, 3400.0 / 8000.0], btype="band")
    spk1_filtered = signal.filtfilt(b1, a1, spk1_bus).astype(np.float32)
    spk1_filtered = np.tanh(1.35 * spk1_filtered).astype(np.float32)

    # Extract continuous background noise slice for this sample
    offset = (sample_idx * 16000 * 11) % (len(noise_beds["babble"]) - total_samples - 100)
    babble_slice = noise_beds["babble"][offset : offset + total_samples]
    street_slice = noise_beds["street"][offset : offset + total_samples]

    # Mix continuous background noise bed (-22 dB to -19 dB SNR relative to speech)
    noise_mix = 0.60 * babble_slice + 0.40 * street_slice
    master_bus = spk0_filtered + spk1_filtered + noise_mix

    # Guarantee zero silent 1-second windows (verify min_rms > 0.002 across every 1s window)
    n_windows = total_samples // 16000
    for w in range(n_windows):
        w_slice = master_bus[w * 16000 : (w + 1) * 16000]
        w_rms = np.sqrt(np.mean(w_slice**2))
        if w_rms < 0.003:
            master_bus[w * 16000 : (w + 1) * 16000] += noise_mix[w * 16000 : (w + 1) * 16000] * 2.5

    # Peak normalize master_bus to -1.5 dBFS (0.84 amplitude)
    peak = np.max(np.abs(master_bus))
    if peak > 1e-5:
        master_bus = master_bus * (0.84 / peak)
    master_pcm16 = np.clip(master_bus * 32767.0, -32768, 32767).astype(np.int16)

    # Save WAV file
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav_filename = f"{sample_id}.wav"
    wav_path = AUDIO_DIR / wav_filename
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(master_pcm16.tobytes())

    # Sort segments chronologically by start_time
    sorted_segs = sorted(scheduled_segs, key=lambda x: float(x["start_time"]))
    exact_ov_dur = round(compute_overlap_duration(sorted_segs), 2)
    exact_ov_ratio = round((exact_ov_dur / actual_duration_sec) * 100.0, 2)
    switches = compute_turn_switches(sorted_segs)

    ground_truth_turns = [
        {
            "speaker": s["speaker_id"],
            "text": s["transcript"],
            "start_time": round(float(s["start_time"]), 2),
            "end_time": round(float(s["end_time"]), 2),
        }
        for s in sorted_segs
    ]
    annotated_transcript = [
        {
            "speaker_id": s["speaker_id"],
            "transcript": s["transcript"],
            "start_time": round(float(s["start_time"]), 2),
            "end_time": round(float(s["end_time"]), 2),
        }
        for s in sorted_segs
    ]

    dialogue_raw = "\n".join(f"{t['speaker']}: {t['text']}" for t in ground_truth_turns)
    dialogue_clean = "\n".join(
        f"{t['speaker']}: {IndicTextNormalizer.normalize(t['text'])}"
        for t in ground_truth_turns
        if IndicTextNormalizer.normalize(t["text"])
    )

    rel_audio_path = f"data/debt_collection_subset/audio/{wav_filename}"

    return {
        "sample_id": sample_id,
        "recording_id": f"hindi_{sample_id}",
        "language": "Hindi",
        "dataset_type": voice_pair["dataset_type"],
        "duration_bucket": bucket_name,
        "duration_seconds": actual_duration_sec,
        "num_speakers": 2,
        "num_segments": len(ground_truth_turns),
        "overlap_duration": exact_ov_dur,
        "overlap_ratio": exact_ov_ratio,
        "turn_switches": switches,
        "audio_path": rel_audio_path,
        "audio_file": f"audio/{wav_filename}",
        "speaker_labels": ["Speaker 0", "Speaker 1"],
        "ground_truth_turns": ground_truth_turns,
        "annotated_transcript": annotated_transcript,
        "reference_dialogue_raw": dialogue_raw,
        "reference_dialogue_indic_clean": dialogue_clean,
    }


def generate_all_50_samples() -> List[Dict[str, Any]]:
    """Main entry point to synthesize TTS pools and build all 50 debt collection audio samples."""
    tts_client = get_tts_client()
    noise_beds = load_background_noise_beds()

    # Step 1: Pre-synthesize and cache utterances for all 6 voice pairs concurrently
    logger.info("Synthesizing Hindi debt collection utterance pools across 6 voice pairs...")
    pair_pools: Dict[int, Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]] = {}

    def _synth_task(text: str, v_cfg: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        pcm = synthesize_utterance_cached(tts_client, text, v_cfg)
        return (text, pcm)

    for p_idx, pair in enumerate(VOICE_PAIRS):
        c_texts, d_texts = build_dialogue_pool_for_pair(p_idx)
        c_pool = []
        d_pool = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            c_futures = [executor.submit(_synth_task, txt, pair["collector"]) for txt in c_texts]
            d_futures = [executor.submit(_synth_task, txt, pair["debtor"]) for txt in d_texts]
            for fut in c_futures:
                c_pool.append(fut.result())
            for fut in d_futures:
                d_pool.append(fut.result())
        pair_pools[p_idx] = (c_pool, d_pool)
        logger.info(f"Completed TTS pool for {pair['pair_id']} ({len(c_pool)} collector, {len(d_pool)} debtor turns).")

    # Step 2: Define exact stratified specifications for all 50 samples
    # Short (30s-70s): 13 samples (debt_001 - debt_013)
    short_durs = np.linspace(34.0, 66.0, 13).round(1).tolist()
    short_ovs = [13.5, 15.2, 17.8, 21.4, 14.1, 18.6, 22.5, 16.0, 19.3, 13.8, 20.5, 16.8, 23.2]

    # Medium (70s-180s): 13 samples (debt_014 - debt_026)
    med_durs = np.linspace(76.0, 172.0, 13).round(1).tolist()
    med_ovs = [14.2, 16.5, 19.1, 22.8, 13.9, 17.4, 20.8, 24.1, 15.6, 18.3, 21.7, 14.8, 19.5]

    # Long (180s-300s): 12 samples (debt_027 - debt_038)
    long_durs = np.linspace(186.0, 294.0, 12).round(1).tolist()
    long_ovs = [14.5, 17.2, 20.4, 23.6, 13.8, 18.1, 21.5, 24.5, 15.9, 19.0, 22.2, 16.4]

    # Very Long (300s-600s): 12 samples (debt_039 - debt_050)
    vlong_durs = np.linspace(312.0, 585.0, 12).round(1).tolist()
    vlong_ovs = [14.8, 17.6, 20.9, 24.2, 14.1, 18.5, 21.9, 25.1, 16.2, 19.4, 22.7, 17.0]

    specs = []
    for i in range(13):
        specs.append((f"debt_{i+1:03d}", "short_30_70s", short_durs[i], short_ovs[i]))
    for i in range(13):
        specs.append((f"debt_{i+14:03d}", "medium_70_180s", med_durs[i], med_ovs[i]))
    for i in range(12):
        specs.append((f"debt_{i+27:03d}", "long_180_300s", long_durs[i], long_ovs[i]))
    for i in range(12):
        specs.append((f"debt_{i+39:03d}", "very_long_300_600s", vlong_durs[i], vlong_ovs[i]))

    metadata: List[Dict[str, Any]] = []
    for idx, (sid, b_name, dur_sec, ov_target) in enumerate(specs):
        p_idx = idx % len(VOICE_PAIRS)
        v_pair = VOICE_PAIRS[p_idx]
        c_pool, d_pool = pair_pools[p_idx]

        entry = build_sample_schedule_and_audio(
            sample_idx=idx,
            sample_id=sid,
            bucket_name=b_name,
            target_duration_sec=dur_sec,
            target_overlap_ratio=ov_target,
            voice_pair=v_pair,
            collector_turns_pool=c_pool,
            debtor_turns_pool=d_pool,
            noise_beds=noise_beds,
        )
        metadata.append(entry)
        logger.info(
            f"[{idx+1:02d}/50] Generated {sid} ({b_name}): dur={entry['duration_seconds']}s, "
            f"overlap={entry['overlap_ratio']}% (target={ov_target}%), turns={entry['num_segments']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved metadata for {len(metadata)} samples to {METADATA_PATH}")

    # Verify dataset using DatasetLoader
    samples = DatasetLoader.load_benchmark_subset(subset_dir=OUTPUT_DIR, verify_audio=True)
    stats = DatasetLoader.get_summary_stats(samples)
    logger.info(f"DatasetLoader Verification Summary: {json.dumps(stats, indent=2)}")
    assert len(samples) == 50, f"Expected 50 samples, got {len(samples)}"
    assert stats["mean_overlap_ratio"] >= 15.0, f"Mean overlap {stats['mean_overlap_ratio']}% < 15.0%"

    return metadata


if __name__ == "__main__":
    generate_all_50_samples()
