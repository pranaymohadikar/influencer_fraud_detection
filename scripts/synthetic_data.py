import pandas as pd
import numpy as np
import logging

# logger config

logging.basicConfig(filename = "logs\data_collection.log", level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    logging.info("creating a synthetic data")
    np.random.seed(42)
    num_influencer = 100000
    logging.info('generating influencer ID, followers, average likes and comments')
    data = {
        
    "influencer_ID": [f'inf{i}' for i in range(1, num_influencer+1)],
    "followers": np.random.randint(1000, 1000000, size = num_influencer),
    "avg_likes": np.random.randint(10, 100000, size = num_influencer),
    'avg_comments': np.random.randint(5, 5000, size = num_influencer),   
}
except Exception as e:
    logging.error(f'error during generation')
    


# data['engagement_rate'] = (data['avg_likes']+data['avg_comments'])/data['followers']

# data['follower_growth_rate'] = np.random.uniform(-0.1, 0.5, size = num_influencer)
# data['suspicious_comment_ratio'] = np.random.uniform(0,0.3, size = num_influencer)

#fraud = 1, genuine = 0
#fraud = low engagement rate or high comment ratio
#data['fraudulent'] = ((data['engagement_rate'] < 0.1) | (data['suspicious_comment_ratio']>0.2)).astype(int)

df = pd.DataFrame(data)
#print(df.head())


output = r'E:\KaaM\Influencer_Fraud_Detection\data\influencer_fraud_detection_synthetic_data.csv'
df.to_csv(output, index = False)

