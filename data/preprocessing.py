import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

website_phishing = fetch_ucirepo(id=379)
X = website_phishing.data.features
y = website_phishing.data.targets