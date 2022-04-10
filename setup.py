import setuptools

def readme():
    with open("./README.md", encoding = "utf-8") as file:
        content = file.read()
    return content

setuptools.setup(name = "tfdet",
                 version = "1.0.0",
                 author = "Hyungjin Kim",
                 author_email = "flslzk@gmail.com",
                 url = "https://github.com/Burf/tfdetection",
                 description = "Detection Toolbox for Tensorflow2",
                 long_description = readme(),
                 long_description_content_type = "text/markdown",
                 license = "Apache License 2.0",
                 install_requires = ["tensorflow>=2.4", "keras"],
                 packages = setuptools.find_packages(),
                 zip_safe = False)
