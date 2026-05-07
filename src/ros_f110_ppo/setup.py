from setuptools import setup

package_name = 'ros_f110_ppo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='feras',
    maintainer_email='ferasbabikerjan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "train_ros_f110_ppo = ros_f110_ppo.train_ros_f110_ppo:main",
            "run_ros_f110_ppo = ros_f110_ppo.run_ros_f110_ppo:main",
            "run_diffusion_f110 = ros_f110_ppo.diffusion_runner:main",
        ],
    },
)
