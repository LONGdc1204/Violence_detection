import torch
import torch.nn as nn
from ResNets_3D_CNN.models.resnet import ResNet
from torch.utils.data import DataLoader

def training_model(num_epochs: int, 
                    model_instance: ResNet, 
                    train_dl: DataLoader, 
                    val_dl: DataLoader,
                    device: torch.device,
                    learning_rate: float,
                    weight_decay = 0.00005):

    # tạo hàm mất mát
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model_instance.parameters(), lr = learning_rate, weight_decay = weight_decay, amsgrad = True)
    
    # training accuracy 
    train_acc_total = []
    # training loss 
    train_loss_total = []
    # validation accuracy 
    val_acc_total = []
    # validation loss 
    val_loss_total = []
    # dữ liệu cho ma trận nhầm lẫn
    cf_matrix_data = []   # [(predict),(label)]
    for epoch in range(num_epochs):
        # trainding accuracy trong từng epoch
        training_corrects = 0
        # training loss trong từng epoch
        training_loss = 0
        # đặt mô hình vào trạng thái train
        model_instance.train()
        val_output_data = []
        for order, data in enumerate(train_dl):
            # lấy dữ liệu và label từ dataloader, cast vào thiết bị đang chạy
            inputs, labels = data[0].to(device), data[1].to(device)
            # xoá gradient
            optimizer.zero_grad()
            # feed forward dữ liệu qua mô hình
            y_pred = model_instance(inputs)
            # tính hàm mất mát
            loss = criterion(y_pred, labels)
            training_loss += loss.item()
            # thực hiện backprobpagation
            loss.backward()
            optimizer.step()

            # Tính toán độ chính xác
            class_pred = torch.argmax(y_pred,1)
            training_corrects += torch.eq(class_pred, labels).sum()/labels.shape[0]
            # xoá tensor
            del inputs
            torch.cuda.empty_cache()
        # Thông tin tổng hợp cho training
        in_batch_train_acc = training_corrects/len(train_dl)
        in_batch_train_loss = training_loss/len(train_dl)
        print("Epoch: ",epoch,", training accuracy: ",in_batch_train_acc,", traing loss: ", in_batch_train_loss)

        # append dữ liệu vào dữ liệu tổng
        train_acc_total.append(in_batch_train_acc)
        train_loss_total.append(in_batch_train_loss)


        ###################### thực hiện validation
        validation_acc = 0
        validation_loss = 0
        model_instance.eval()
        with torch.no_grad():
            print("Run validation")
            for val_order,val_data in enumerate(val_dl):
                # lấy dữ liệu và label từ dataloader, cast vào thiết bị đang chạy
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)

                # feed forward dữ liệu qua mô hình
                val_pred = model_instance(val_inputs)
                # tính hàm mất mát trong validation
                val_loss = criterion(val_pred, val_labels)
                validation_loss += val_loss.item()
                # tính độ chính xác trong validation
                val_class_pred = torch.argmax(val_pred,1)
                validation_acc += torch.eq(val_class_pred, val_labels).sum()/val_labels.shape[0]

                # lưu dữ liệu vào ma trận nhầm lẫn
                val_output_data.append( (val_class_pred,val_labels) )

                del val_inputs
                torch.cuda.empty_cache()

        # Thông tin tổng hợp cho training
        in_batch_val_acc = validation_acc/len(val_dl)
        in_batch_val_loss = validation_loss/len(val_dl)
        print("Epoch: ",epoch,", validation accuracy: ",in_batch_val_acc,", validation loss: ", in_batch_val_loss)
        # append validation metric to master
        val_acc_total.append(in_batch_val_acc)
        val_loss_total.append(in_batch_val_loss)

        # append dữ liệu vào dữ liệu tổng
        cf_matrix_data.append(val_output_data)

    return train_acc_total, train_loss_total, val_acc_total, val_loss_total, cf_matrix_data