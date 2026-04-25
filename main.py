from src.models.baseline import baseline_model
from src.models.time_of_day import hourly_model


def main():
    
    print("----------- BASELINE MODEL -----------")
    baseline_model()
    
    print("----------- HOURLY MODEL -----------")
    hourly_model()


if __name__=="__main__":
    main()