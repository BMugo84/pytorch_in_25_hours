
"""
Trains a pytorch image classification model using device agnostic code
"""
import os
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

# setup hyper-parameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# create transforms
data_transform = transforms.Compose([
                                      transforms.Resize((64,64)),
                                      transforms.ToTensor()
])

# create dataloaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                            test_dir,
                                                                            data_transform,
                                                                            BATCH_SIZE)

# build model
model_1 = model_builder.TinyVGG(input_shape=3,
                                hidden_units=10,
                                output_shape=len(class_names)).to(device)

# setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=LEARNING_RATE)

# start the timer
start_time = timer()

# train model
engine.train(model=model_1,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS,
            device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# save the model
utils.save_model(model=model_1,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
