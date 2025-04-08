import pandas as pd #type: ignore
import torch
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from ucimlrepo import fetch_ucirepo #type: ignore

website_phishing = fetch_ucirepo(id=379)
X = website_phishing.data.features
y = website_phishing.data.targets