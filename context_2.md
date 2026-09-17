We tried the approach Lex shared on a few examples; it's not yielding good results. Here is our hypothesis: 

1. Poor performance is directly related to length of an audio file -> accuracy with Gemini 3.5 Flash lite exponentially decreases with increase in size of the call, Here Gemini 3.5 Flash performs really well. I think the model is suffering from a long context attention drop.
2. Conversation vs. Argumentative: For argumentative calls, when people talk over each other with no pause, G 3.5 Flash lite performance drops substantially. 

Unfortunately, the customer is testing with a dataset containing large, argumentative files—which they believe is the critical area—but these files represent less than 10% of all calls. We are suggesting they use the G 3.5 Flash model for those kinds of calls. 

Here are the next steps for us

1. Create a good representation of a dataset with audio files of different lengths. 
2. Run all 3 approaches to determine which works the best and to find the audio length cutoff above which the G 3.5 Flash Lite model's performance degrades.
3. Build a robust routing layer to route calls to the appropriate model based on call characteristics.

@Lex Ha : On the Sarvam dataset, could you find the relationship between the length of the audio file and the accuracy of speaker diarization? I am not sure how easily we can filter the dataset for argumentative calls, I think in step 1 you are also creating call characteristics; perhaps we can use those to filter those records and see how the Lite model performs on that subset.

I know it's difficult to do anything without the right representative dataset but we are working with the customer to find a way to fix this. I really appreciate all the help you are providing. Your experiments are really helping us with insights.
