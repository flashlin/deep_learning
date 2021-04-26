import xml.etree.ElementTree as ET


class AnnotationXml:
    xml_list = []

    def __init__(self, annotation_xml_file_path):
        self.xml_file_path = annotation_xml_file_path
        tree = ET.parse(annotation_xml_file_path)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        for member in root.findall('object'):
            label_name = member[0].text
            xml_bndbox = member.find("bndbox")
            x_min = int(xml_bndbox.find("xmin").text)
            y_min = int(xml_bndbox.find("ymin").text)
            x_max = int(xml_bndbox.find("xmax").text)
            y_max = int(xml_bndbox.find("ymax").text)
            self.xml_list.append((
                filename,
                width, height,
                label_name,
                (x_min, y_min, x_max, y_max)
            ))

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        return self.xml_list[index]
