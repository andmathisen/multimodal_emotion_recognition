import matplotlib.pyplot as plt
import mlxtend
import matplotlib
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

print('matplotlib version', matplotlib.__version__)
print('mlxtend version', mlxtend.__version__)

cm = np.array([[10,	5,	0,	0	,5	,1,	11],
                [1,	75,	4	,4	,1	,2	,3	],
                [1,	1,	50	,1,	1,	0,	7	],
                [0,	2,	0	,88,	0,	0,0	],
                [0,	3,	4	,0,	43,	3,	16],
                [1,	7,	1	,0	,3	,48,	30],
                [2,	5,	5,	0	,7,	13,	58]])
class_names = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
fig, ax = plot_confusion_matrix(conf_mat=cm,class_names=class_names)
fig.tight_layout()
plt.savefig("cm_p4_mer2.png",transparent=True)


