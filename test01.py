import os

dir_path = r"D:\Code\ML\images\Mywork3\card_database_no_compress\prizm\21-22"

for name in os.listdir(dir_path):
    dir_name = os.path.join(dir_path, name)
    new_name = os.path.join(dir_path, name.split(' ')[0])

    os.rename(dir_name, new_name)
    print(name, ' ====>> ', name.split(' ')[0])
