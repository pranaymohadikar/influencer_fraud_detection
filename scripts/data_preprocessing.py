import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime


# Logging configuration
logging.basicConfig(filename='logs/feature_engineering.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')