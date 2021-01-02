import os
from traceback import print_tb

from PIL import Image, ImageEnhance
from PIL import ImageOps

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_path = "C:/Users/megag/PycharmProjects/TF-IA/resources/train"


def augment_data():
    image_list = []
    for root, dirs, files in os.walk(train_path):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(path, file)
            image_list.append((path, file))

    # print(image_list)

    for path, name in image_list:
        pt = path[0] + "/" + path[1]
        try:
            print(pt + "/" + name)
            im = Image.open(pt + "/" + name)
            # im_output = ImageOps.mirror(im)
            # im_output.save(pt+"/mirror_"+name)

            enhancer = ImageEnhance.Brightness(im)
            im_output = enhancer.enhance(0.3)
            im_output.save(pt + "/bright03_" + name)

            enhancer = ImageEnhance.Brightness(im)
            im_output = enhancer.enhance(0.5)
            im_output.save(pt + "/bright05_" + name)

            enhancer = ImageEnhance.Brightness(im)
            im_output = enhancer.enhance(0.8)
            im_output.save(pt + "/bright08_" + name)

            enhancer = ImageEnhance.Brightness(im)
            im_output = enhancer.enhance(1.3)
            im_output.save(pt + "/bright13_" + name)
        except:
            print("ERROR")
            print(pt + "/" + name)


def correct_data():
    import os
    from PIL import Image
    folder_path = train_path
    extensions = []
    for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            print('** Path: {}  **'.format(file_path))
            try:
                im = Image.open(file_path)
            except:
                os.remove(file_path)
                print("Deleted", file_path)
            rgb_im = im.convert('RGB')
            if filee.split('.')[1] not in extensions:
                extensions.append(filee.split('.')[1])



correct_data ()