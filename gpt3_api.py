import openai

from constants import OPEN_AI_API_KEY

openai.api_key = OPEN_AI_API_KEY

sample = "Abraham Lincoln ; February 12, 1809 April 15, 1865 was an American lawyer and statesman who served as the 16th president of the United States from 1861 until his assassination in 1865. Lincoln led the nation through the American Civil War and succeeded in preserving the Union, abolishing slavery, bolstering the federal government, and modernizing the U.S. economy. Lincoln was born into poverty in a log cabin in Kentucky and was raised on the frontier, primarily in Indiana. He was self-educated and became a lawyer, Whig Party leader, Illinois state legislator, and U.S. Congressman from Illinois. In 1849, he returned to his law practice but became vexed by the opening of additional lands to slavery as a result of the Kansas–Nebraska Act of 1854. He reentered politics in 1854, becoming a leader in the new Republican Party, and he reached a national audience in the 1858 Senate campaign debates against Stephen Douglas. Lincoln ran for President in 1860, sweeping the North to gain victory. Pro-slavery elements in the South viewed his success as a threat to slavery, and Southern states began seceding from the Union. To secure its independence, the new Confederate States fired on Fort Sumter, a U.S. fort in South Carolina, and Lincoln called up forces to suppress the rebellion and restore the Union. Lincoln, a moderate Republican, had to navigate a contentious array of factions with friends and opponents from both the Democratic and Republican parties. His allies, the War Democrats and the Radical Republicans, demanded harsh treatment of the Southern Confederates. Anti-war Democrats (called "Copperheads") despised Lincoln, and irreconcilable pro-Confederate elements plotted his assassination. He managed the factions by exploiting their mutual enmity, carefully distributing political patronage, and by appealing to the American people. His Gettysburg Address appealed to nationalistic, republican, egalitarian, libertarian, and democratic sentiments. Lincoln supervised the strategy and tactics in the war effort, including the selection of generals, and implemented a naval blockade of the South's trade. He suspended habeas corpus in Maryland, and he averted British intervention by defusing the Trent Affair. He engineered the end to slavery with his Emancipation Proclamation, including his order that the Army and Navy liberate, protect, and recruit former slaves. He also encouraged border states to outlaw slavery, and promoted the Thirteenth Amendment to the United States Constitution, which outlawed slavery across the country. Lincoln managed his own successful re-election campaign. He sought to heal the war-torn nation through reconciliation. On April 14, 1865, just days after the war's end at Appomattox, he was attending a play at Ford's Theatre in Washington, D.C., with his wife Mary when he was fatally shot by Confederate sympathizer John Wilkes Booth. Lincoln is remembered as a martyr and hero of the United States and is often ranked as the greatest president in American history."

result = openai.Answer.create(
    search_model="ada", 
    model="curie", 
    question='test', 
    documents='testing', 
    examples_context=sample, 
    examples=[["Who was the 16th president of the United States?", "Abraham Lincoln"], ["Through what major war did Abraham Lincoln serve as United States president?", "American Civil War"], ["What year was Abraham Lincoln killed?", "1865"]], 
    max_rerank=3,
    max_tokens=5,
    stop=["\n", "<|endoftext|>"]
)
