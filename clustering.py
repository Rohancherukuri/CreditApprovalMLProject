import warnings 
import logging
import joblib
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
PATH = "/home/rohanoxob/MachineLearning/CreditApprovalProject/logs/clustering.log"
#logging.basicConfig(filename = PATH,level = logging.INFO,format = LOG_FORMAT,filemode = "w")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class KMeansClustering:
    # This class is to generate the data into clusters before trainig
    def __init__(self,data):
        self.data = data
        # Data Preprocessing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
            
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2) # Using PCA for dimensionality reduction
        pca.fit(scaled_data)
        pca_features = pca.transform(scaled_data)
            
        self.pca_features = pca_features
        logger.info("Inside KMeansClustering class constructor")
    
    def elbow_plot(self):
        """
            Method Name: elbow_plot
            Description: Saves the plot to decide the optimum number of clusters to the file.
            Output: A png file saved to the directory
            On Failure: Raise Exception
        """
        logger.info("Inside the elbow_plot function in KMeansClustering class")
        try:
            data = self.data
            k_range = range(1,11)
            wcss = [] # Initialising the empty list
            for k in k_range:
                km = KMeans(n_clusters = k,init = "k-means++",random_state = 100,max_iter = 1000)
                km.fit(self.pca_features) # Fitting the data to the kmeans algorithm
                wcss.append(km.inertia_)
            plt.title("K vs WCSS (ElbowPlot)")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            plt.plot(k_range,wcss) # Creating the graph between k vs WCSS 
            plt.savefig("Graphs/Elbow_plot.png") # Saving the elbow plot
            plt.close()
            # Finding the value of optimal cluster 
            kn = KneeLocator(k_range,wcss,curve = "convex",direction = "decreasing")
            logger.info("The optimum number of clusters are: {}, Exited the elbow_plot function of KMeansClustering class".format(kn.knee))
            return kn.knee
        
        except Exception as e:
            logger.warning("Exception has occured in elbow_plot method of the KMeansClustering class. Exception message: "+str(e))
            logger.warning("Finding the number of clusters failed. Exitted the elbow_plot function of the KMeanClustering class")
            raise Exception()            
            
                
    
    def create_clusters(self,k_clusters):
        """
            Method Name: create_clusters
            Description: Creates a new dataframe consisting og the clusters information
            Output: A dataframe with cluster column.
            On Failure: Raise Exception
        """
        logger.info("Inside the create_clusters function of the KMeanClustering class")
        
        try:
            data = self.data    
            
            model_path = "/home/rohanoxob/MachineLearning/CreditApprovalProject/models/"
           
            kmeans = KMeans(n_clusters = k_clusters,init = "k-means++",random_state = 100,max_iter = 1000)
            y_predicted = kmeans.fit_predict(self.pca_features)
            joblib.dump(kmeans,model_path +"model.sav") # Saving the kmeans model
            data["Cluster"] = y_predicted # Creating a new cluster column for storing the cluster information
            logger.info("Successfully created {} clusters. Exited the create_clusters function of the KMeansClustering class ".format(k_clusters))
            
            try:
                # Creating a cluster map
                plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1], c = y_predicted, cmap = 'plasma')
                plt.title("Cluster Map")
                plt.xlabel('pca1')
                plt.ylabel('pca2')
                plt.savefig("Graphs/Cluster_Map.png") # Saving the cluster map
                plt.close()
                logger.info("Successfully created the cluster map")
            except:
                logger.warning("Failed to create Cluster Map")
                raise Exception()
             
            return data
        
        except Exception as e:
            logger.warning("Exception has occured in create in create_clusters method of the KMeansClustering class. Exception message: "+str(e))
            logger.warning("Fitting the data to clusters failed. Exited the create_clusters function of KMeansClustering class")
            raise Exception()
        
# This is for testing purposes         
"""if __name__ == "__main__":
    try:
        DATA_PATH = "/home/rohanoxob/MachineLearning/CreditApprovalProject/data.csv"
        df = pd.read_csv("data.csv")
        df_new = df.drop(['Unnamed: 0', 'ID'], axis='columns')
        k = KMeansClustering(df_new)
        clus = k.elbow_plot()
        print(clus)
        data = k.create_clusters(clus)
        print(data.head())
    except Exception as e:
        print("Sorry there was an error in your code: "+str(e))
"""