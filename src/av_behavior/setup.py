from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'av_behavior'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/av_behavior']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py') if os.path.exists('launch') else []),
        ('share/' + package_name + '/config',
            glob('config/*.yaml') if os.path.exists('config') else []),
        ('lib/' + package_name,
            glob('lib/' + package_name + '/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='victor',
    maintainer_email='victor@email.com',
    description='av_behavior package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'behavior_node = av_behavior.behavior_node:main',
        ],
    },
)
