from setuptools import setup


if __name__ == '__main__':
    with open('README.md', 'r', encoding='utf8') as f:
        long_description = f.read()

    setup(
        name='bleu-mp',
        version='0.0.1',
        description='Multi-process bleu evaluation.',
        author='onesixth',
        author_email='noexist@noexist.noexist',
        url='https://github.com/One-sixth/bleu-mp',
        install_requires=[],
        packages=['bleu_mp'],
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: Microsoft :: Windows",
        ],
    )
