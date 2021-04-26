import os


class ClassesDict:
    dict = {}
    names = []

    def add_classes_name(self, name):
        if name in self.dict:
            return self.dict[name]
        self.names.append(name)
        self.dict[name] = len(self.names)
        return self.dict[name]

    def add_classes_names_by_classification_dir(self, classification_dir):
        for classes_name in os.listdir(classification_dir):
            self.add_classes_name(classes_name)

    def get_classes_name(self, index):
        name = self.names[index]
        return name

    def save(self, save_file_path):
        with open(save_file_path, "w") as f:
            for name in self.names:
                f.write(f"{name}\n")

    def __iter__(self):
        for name, name_id in self.dict:
            yield name, name_id