#!/usr/bin/env python3
import pathlib
import setuptools

# setup.py metadata
setup_location = "." # pathlib.Path(__file__).parents[1]

# package metadata: general descriptors
package_name = "rrBLUP"
package_version = "0.1.0"
package_author = "Robert Z. Shrote"
package_author_email = "shrotero@msu.edu"
package_description = "Python package to fit rrBLUP genomic prediction models"
with open("README.md", "r", encoding = "utf-8") as readme_file:
    package_description_long = readme_file.read()
    package_description_long_type = "text/markdown"

# package metadata: project URLs
package_url = "https://github.com/rzshrote/rrblup"
package_project_url = {
    "Bug Tracker": "https://github.com/rzshrote/rrblup/issues",
}

# package metadata: licensing and classifiers
package_license = "Apache License 2.0"
package_classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
]

# package metadata: installation requirements
package_requirements_python = ">=3.6"
package_requirements_install = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
]

# package metadata: package locations
package_package_directory = {"" : setup_location}
package_packages = setuptools.find_packages(where = setup_location)

# setup the package
setuptools.setup(
    name = package_name,
    version = package_version,
    author = package_author,
    author_email = package_author_email,
    description = package_description,
    long_description = package_description_long,
    long_description_content_type = package_description_long_type,
    url = package_url,
    project_urls = package_project_url,
    license = package_license,
    classifiers = package_classifiers,
    package_dir = package_package_directory,
    packages = package_packages,
    python_requires = package_requirements_python,
    install_requires = package_requirements_install
)
