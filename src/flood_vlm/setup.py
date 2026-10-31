import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'flood_vlm'


def tree(src):
    """data_files entries that install a directory tree under share/."""
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*')),
    ] + tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahtishamsudheer',
    maintainer_email='ahtishamsudheer@gmail.com',
    description='FloodScout-VLM: language-guided UAV flood-relief scout with a CPU-only VLM',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'scene_gen = flood_vlm.scene_gen:main',
            'water_controller = flood_vlm.water_controller:main',
            'mission_node = flood_vlm.mission_node:main',
            'floodsim_vqa_gen = flood_vlm.eval.floodsim_vqa_gen:main',
            'floodnet_prep = flood_vlm.eval.floodnet_prep:main',
            'run_vqa = flood_vlm.eval.run_vqa:main',
            'mission_eval = flood_vlm.eval.mission_eval:main',
        ],
    },
)
