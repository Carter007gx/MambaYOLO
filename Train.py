from ultralytics import YOLO



if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8l.yaml")  # build a new model from scratch

    # Use the model
    model.train(data="SHWD.yaml", epochs=300, batch=16)  # train the model

    model.val(data="SHWD.yaml", split="test")  # evaluate model performance on the validation set


