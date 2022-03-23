# coding=utf-8
from setuptools import setup

setup(
    name="yolox",
    version="0.1.0",
    py_modules=["yolox"],
    install_requires=[
        "Click",
        "depthai-sdk>=1.1.6",
        "depthai>=2.13.0.0",
    ],
    dependency_links=[
        "http://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.python.org/simple",
    ],
    python_requires=">=3",
    entry_points={
        "console_scripts": [
            "yolox = yolox:yolox",
        ],
    },
)
