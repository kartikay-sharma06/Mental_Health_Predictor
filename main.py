
from src.components.data_ingestion import data_ingestion

if __name__ == "__main__":
    
        data_ingestion = data_ingestion()
        train_data_path, test_data_path = (
            data_ingestion.initiate_data_ingestion()
        )

        print("Data ingestion completed successfully.")
        print(f"Training data saved at: {train_data_path}")
        print(f"Testing data saved at: {test_data_path}")

    
