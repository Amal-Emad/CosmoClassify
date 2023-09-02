# CosmoClassify
conveys My vision to categorize and understand celestial objects in the universe using advanced machine learning techniques

# Project Name: CosmoClassify
Cosmo: This part of the name is derived from "cosmos," which is a term denoting the entire universe. It underscores the overarching scope and focus of our project, which is the exploration and study of celestial phenomena, including stars, galaxies, and other celestial objects within the vast expanse of space.

Classify: This part of the name highlights the fundamental function of our project. It represents our mission to organize, categorize, and make sense of the extensive data gathered by the James Webb Space Telescope. Machine learning algorithms are at the core of this process, aiding in the classification and identification of diverse space entities.


# 1. Abstract
In a monumental achievement, space agencies like the European Space Agency and NASA have triumphed in conquering one of the cosmos's most formidable challenges: the successful launch and orbital placement of the awe-inspiring James Webb Space Telescope. As we stand on the cusp of a groundbreaking era in space exploration, this marvel of technology is poised to unveil the secrets of the universe in unparalleled detail.

With the James Webb Space Telescope securely nestled in its designated orbit, it is merely a matter of months before this cutting-edge observatory becomes fully operational. In the not-so-distant future, Earth will be showered with an astounding deluge of celestial images captured from the deepest corners of the cosmos. Prepare to embark on an extraordinary journey as this remarkable telescope opens a window to the vast expanse of the universe, revealing hitherto unseen stars, galaxies, and enigmatic quasars.

However, with this treasure trove of data comes a monumental challenge: the need to make sense of it all. The cosmos is about to unleash an overwhelming influx of new space objects that demand classification and understanding. To tackle this colossal task, the inevitable solution lies in the realm of machine learning.

Harnessing the power of advanced machine learning algorithms, we are poised to revolutionize our understanding of the universe. These algorithms will sift through the vast expanse of data pouring in from the telescope, sorting, categorizing, and revealing the mysteries of the cosmos like never before.

To ensure the accuracy and reliability of this groundbreaking mission, a robust and comprehensive database has been meticulously assembled, housing a wealth of data points. This treasure trove of information will serve as the foundation upon which our machine learning algorithms will stand, helping us unlock the secrets of the universe and redefine our place in the cosmos.

The James Webb Space Telescope is not just a marvel of engineering; it's a gateway to the unknown, a vessel of discovery, and a testament to human ingenuity. Get ready to embark on an incredible journey as we explore the universe like never before, one data point at a time.

# 2. Dataset

For my project, I obtained a dataset from the Kaggle database system, which was originally published by the Sloan Digital Sky Survey and gathered using a ground telescope in New Mexico, USA. This dataset is instrumental in preparing for the massive influx of data expected from the James Webb Telescope.

The dataset contains a substantial one hundred thousand data points, making it highly valuable for my project's needs. However, I decided to exclude certain ID-related features and unique identifiers, such as spec_obj_ID and obj_ID, as they do not contribute to the core analysis. During the data preprocessing stage, I manually selected the most relevant features that will be used throughout the project. These features include:

1. Right Ascension angle (alpha)
2. Declination angle (delta)
3. Ultraviolet filter (u)
4. Green filter (g)
5. Red filter (r)
6. Near Infrared filter (i)
7. Infrared filter (z)
8. Redshift value (redshift)
9. Field ID

I chose these features because they are vital for identifying the class of space objects and understanding their characteristics. While field ID varies among space objects, it also exhibits some repeating instances, which I believe could influence the project's results.

In essence, my project, known as "cosmoclassify" is focused on utilizing this dataset to classify and gain insights into space objects, harnessing the power of machine learning algorithms to make sense of the universe's mysteries.
