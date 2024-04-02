import os


def prepare_dir(file_path):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    # Remove the file name to only keep the directory path.
    dir_path = '/'.join(file_path.split('/')[:-1])
    # Try to create the directory. Will have no effect if the directory already exists.
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass