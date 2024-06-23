import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model
import xlwt

book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel

sheet1 = book.add_sheet(u'预测结果', cell_overwrite_ok=True)

# 向表中添加数据
sheet1.write(0, 0, '文件名')
sheet1.write(0, 1, '预期结果')
sheet1.write(0, 2, '预测结果')


def traverse_folder(folder_path):
    list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            list.append(abs_file_path)

    return list


def action_img(file_path):
    device = torch.device("cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = file_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=3, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

    #print(print_res)
    '''plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        plt.show()'''

    return print_res



def main():

    img_clinic_path = r"D:\train\病历"
    img_drug_path = r"D:\train\药品"
    mg_clinic_file_list = traverse_folder(img_clinic_path)
    img_drug_file_list = traverse_folder(img_drug_path)

    index=1

    for item in mg_clinic_file_list:
        sheet1.write(index, 0, item)
        sheet1.write(index, 1, "病历")
        sheet1.write(index, 2, action_img(item))
        index+=1

    for item in img_drug_file_list:
        sheet1.write(index, 0, item)
        sheet1.write(index, 1, "药品")
        sheet1.write(index, 2, action_img(item))
        index += 1

    book.save('.\预测结果.xlsx')





if __name__ == '__main__':
    main()