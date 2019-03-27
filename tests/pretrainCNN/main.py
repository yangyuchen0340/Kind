from Utils.Feature_extractor import Feature_extractor
from Utils.Clusterer import Clusterer
import os

dataset              = "coil-100"
cnn_architecture     = "vgg19"
layer 				 = "fc2"
clustering_algorithm = "SR"
metric				 = "both"



if os.path.exists("./Data/%s/" % dataset + "Features/%s_%s" % (cnn_architecture, layer)):
    fe = Feature_extractor(dataset, cnn_architecture, layer)
    fe.extract_and_save_features()
cl = Clusterer(dataset, cnn_architecture, layer, clustering_algorithm,100)
cl.cluster()
predicted_labels = cl.predicted_labels
print("Shape predicted labels: %s" % str(predicted_labels.shape))
cl.evaluate(metric)