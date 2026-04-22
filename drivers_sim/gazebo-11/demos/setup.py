import os

from setuptools import find_packages, setup

package_name = "demos"


def recursive_get_files(domain_dir: str) -> list[tuple[str, list[str]]]:
    target = []
    for root, _, files in os.walk(domain_dir):
        files_path = [os.path.join(root, f) for f in files]
        if files_path:
            # 目标路径：share/${package_name}/${domain_dir}/相对路径
            dest = os.path.join("share", package_name, root)
            target.append((dest, files_path))
    return target


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *recursive_get_files("launch"),
        *recursive_get_files("config"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            'pose_initialization = demos.init_pose:main'
        ],
    },
)
