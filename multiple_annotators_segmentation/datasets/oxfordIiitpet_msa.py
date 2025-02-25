import subprocess

def OxfordIiitPet_Multiple_Synthetic_Annotators():
    """
    Downloads and prepares the Oxford-IIIT Pet dataset with synthetic annotators.

    Notes:
        - The Kaggle CLI must be installed and configured to authenticate 
          before executing this function. You can set up the Kaggle API 
          by following the instructions at https://www.kaggle.com/docs/api
        - The dataset downloaded by this function contains three synthetic 
          annotators generated by modifying the signal-to-noise ratio of a 
          well-performing UNet model. The annotators are categorized as good (G), 
          normal (N), and bad (B). Each scenario, such as G_G_G or G_N_B, 
          represents different combinations of annotator qualities.
        - The dataset is divided into 1024 training images, 256 validation images, 
          and 256 evaluation images. Each image includes segmentation masks 
          corresponding to each annotator, with distinct channels for background 
          (class 0) and pet (class 1).

    Example:
        OxfordIiitPet_Multiple_Synthetic_Annotators()
    """
    # Download Oxford-IIIT Pet dataset with synthetic annotators
    subprocess.run(["kaggle", "datasets", "download", "-d", "lucasiturriago/oxford-pets/9"])
    
    # Extract the downloaded dataset files
    subprocess.run(["unzip", "-q", "oxford-pets.zip", "-d", "datasets"])