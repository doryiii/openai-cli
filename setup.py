from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='yaoc',
    version='0.1.3',
    author='Dory',
    author_email='dory@dory.moe',
    description='Yet Another OpenAI-compatible CLI',
    license='GPL-3.0-only',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/doryiii/yaoc',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'termcolor',
        'rich',
        'html2text',
    ],
    entry_points={
        'console_scripts': [
            'yaoc=yaoc.openai_cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
