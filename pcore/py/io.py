import os
import re


def confirm_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def sed(file_path, reg_pattern):
    regex = re.compile(r'(\d+)s/(.+)/(.+)/')
    match = regex.search(reg_pattern)
    if not match:
        raise Exception("sid pattern error")

    '212s/255/18/'
    line_num = int(match.group(1)) - 1
    search_pattern = match.group(2)
    replace_txt = match.group(3)

    # info(f"sed {line_num} {search_pattern} {replace_txt}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        line = lines[line_num]
        text_after = re.sub(search_pattern, replace_txt, line)
        lines[line_num] = text_after
        # for line in lines:
        #     f.write(re.sub(reg_pattern, txt, line))
        f.write("".join(lines))


def get_folder_list(the_dir):
    if not os.path.exists(the_dir):
        return []
    return [f"{the_dir}/{name}" for name in os.listdir(the_dir)
            if os.path.isdir(os.path.join(the_dir, name))
            ]
