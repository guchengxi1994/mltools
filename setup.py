from setuptools import find_packages, setup

setup(
    name="mltools",
    version="0.1.2",  # 与 __init__.py 保持一致
    description="A collection of data augmentation and label processing tools.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="xiaoshuyui",
    author_email="guchengxi1994@qq.com",
    url="https://github.com/guchengxi1994/mltools",  # 修改为你的实际仓库
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
