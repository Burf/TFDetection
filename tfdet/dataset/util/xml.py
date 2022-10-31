import os
import xml.etree.ElementTree

def xml2dict(data, attrib_prf = "@", unknown_text = "#text"):
    if not isinstance(data, str):
        raise Exception("Type incorrect")
    
    if os.path.exists(data):
        element = xml.etree.ElementTree.parse(data).getroot()
    else:
        element = xml.etree.ElementTree.fromstring(data)
    
    result = {}
    def parse(element, result):
        key = element.tag
        value = element.attrib.copy()
        for k in list(value.keys()):
            value[attrib_prf + k] = value.pop(k)
            
        childs = element.getchildren() if hasattr(element, "getchildren") else list(element)
        for child in childs:
            parse(child, value)
            
        if not childs:
            if not value:
                value = element.text
            elif element.text is not None:
                value[unknown_text] = element.text
        
        if key in result.keys():
            if not isinstance(result[key], list):
                result[key] = [result[key]]
            result[key].append(value)
        else:
            result[key] = value
    parse(element, result)
    return result

def dict2xml(data, save_path = None):
    key, val = list(data.items())[0]
    root = xml.etree.ElementTree.Element(key)
    
    def push(data, element):
        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(v, list):
                    v = [v]
                for _v in v:
                    push(_v, xml.etree.ElementTree.SubElement(element, k))
        else:
            element.text = data
    push(val, root)
    
    if isinstance(save_path, str):
        tree = xml.etree.ElementTree.ElementTree(root)
        tree.write(save_path, "utf-8")
    return root
