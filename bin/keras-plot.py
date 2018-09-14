# libraries
import utils.file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Variables
model_name = '180914-lstm-128-presumption-test-results'
json_path = f'../results/{model_name}.json'

# Data
data = utils.file.read_json(json_path)
df = pd.DataFrame(data)

# multiple line plot
plt.title(model_name)
plt.plot('losses', data=df, color='#FF9E9D', linewidth=2)
plt.plot('accuracies', data=df, color='#7FC7AF', linewidth=2)
plt.plot('val_losses', data=df, color='#FF3D7F', linewidth=2)
plt.plot('val_accuracies', data=df, color='#3FB8AF', linewidth=2)
plt.plot('val_f1s', data=df, color='#F8CA00', linewidth=2)

plt.legend()
plt.show()
