import matplotlib.pyplot as plt
import seaborn as sns
from scipy import fftpack
from snorkel.labeling import labeling_function

def PlotConfMat(cm, labels=[]):
    import seaborn as sns
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    ax.set_xlabel('Pred');ax.set_ylabel('Real'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
    plt.show()

