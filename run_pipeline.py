from pipelines.training_pipeline import train_pipeline

if __name__=="__main__":
    # Run the pipeline
    train_pipeline(data_path="./data/olist_customers_dataset.csv")