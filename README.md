# uasdeeplearning
uas deeplearning tim 17 gami
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "daun_sayur_buah.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "gpuClass": "standard",
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "x2xyNjR0OFdz"
      },
      "source": [
        "import tensorflow as tensor\n",
        "from tensorflow.keras.optimizers import RMSprop\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "import os\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sn\n",
        "import cv2\n",
        "from random import randint\n",
        "import numpy as np\n",
        "from tensorflow.keras import models, layers\n",
        "import matplotlib.pyplot as plt\n",
        "from PIL import Image\n",
        "import numpy as np\n",
        "from tensorflow.keras.layers.experimental import preprocessing\n",
        "from tensorflow.keras.preprocessing.image import load_img"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install kaggle"
      ],
      "metadata": {
        "id": "deiFlbx_QnXS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "65d5170c-29d3-4298-86f3-d71359fd78d4"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Requirement already satisfied: kaggle in /usr/local/lib/python3.7/dist-packages (1.5.12)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.8.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from kaggle) (2.23.0)\n",
            "Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle) (6.1.2)\n",
            "Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.15.0)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle) (2022.6.15)\n",
            "Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle) (1.24.3)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle) (4.64.0)\n",
            "Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle) (1.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->kaggle) (3.0.4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ],
      "metadata": {
        "id": "WO9LsVVyQpaq",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgZG8gewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwoKICAgICAgbGV0IHBlcmNlbnREb25lID0gZmlsZURhdGEuYnl0ZUxlbmd0aCA9PT0gMCA/CiAgICAgICAgICAxMDAgOgogICAgICAgICAgTWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCk7CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPSBgJHtwZXJjZW50RG9uZX0lIGRvbmVgOwoKICAgIH0gd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCk7CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 90
        },
        "outputId": "1fdd938f-f86f-4626-9101-5119d56b3f51"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-f6b330f5-ec1f-4bf4-be4b-2470069c079b\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-f6b330f5-ec1f-4bf4-be4b-2470069c079b\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Saving kaggle.json to kaggle.json\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'kaggle.json': b'{\"username\":\"kelompok17gami\",\"key\":\"b3f7142e8b6a6b1e18e95af12387920e\"}'}"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!mkdir -p ~/.kaggle\n",
        "!cp kaggle.json ~/.kaggle/\n",
        "!chmod 600 ~/.kaggle/kaggle.json"
      ],
      "metadata": {
        "id": "o1GmTrmeQrUX"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "900AoG_-Ok7m",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9fd9de1f-d205-498c-a296-7ec5f921878a"
      },
      "source": [
        "!kaggle datasets download -d auliapebiani/deeplearningtim17gami"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading deeplearningtim17gami.zip to /content\n",
            " 91% 192M/211M [00:01<00:00, 198MB/s]\n",
            "100% 211M/211M [00:01<00:00, 189MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lefDdtFqOxoq"
      },
      "source": [
        "import zipfile,os\n",
        "zip_lcl = '/content/deeplearningtim17gami.zip'\n",
        "rzip = zipfile.ZipFile(zip_lcl, 'r')\n",
        "rzip.extractall('/content')\n",
        "rzip.close()"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dirbase = '/content/IMAGE CLASIFICATION BUNGA DAUN SAYUR'\n",
        "dir_latih = os.path.join(dirbase, 'TRAIN')\n",
        "dir_val = os.path.join(dirbase, 'VALIDATION')"
      ],
      "metadata": {
        "id": "63a3wezuROs5"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "CLASSES, gems = [], []\n",
        "\n",
        "for root, dirs, files in os.walk(dir_latih):\n",
        "    f = os.path.basename(root) \n",
        "        \n",
        "    if len(files) > 0:\n",
        "        gems.append(len(files))\n",
        "        if f not in CLASSES:\n",
        "            CLASSES.append(f)\n",
        "    \n",
        "gems_count = len(CLASSES)\n",
        "print('{} classes with {} images in total'.format(len(CLASSES), sum(gems)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2uRyXAkJQKnW",
        "outputId": "cc4db279-18da-4471-cafa-2df5152284b1"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3 classes with 240 images in total\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7tIkv-f4QG_q",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "26961fa8-f8d7-4b69-f6df-cec5213eb702"
      },
      "source": [
        "os.listdir(dir_latih)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['DAUN', 'BUNGA', 'SAYUR']"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y49Xm2mOQKWW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c38b852d-74b5-43a4-e47e-44fef1f21795"
      },
      "source": [
        "os.listdir(dir_val)"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['DAUN', 'BUNGA', 'SAYUR']"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "h9k1JJlzUYTE"
      },
      "source": [
        "latih_generator = ImageDataGenerator(\n",
        "    rescale=1./255,\n",
        "    rotation_range=20,\n",
        "    horizontal_flip=True,\n",
        "    shear_range=0.2,\n",
        "    fill_mode='nearest')\n",
        "\n",
        "val_generator = ImageDataGenerator(\n",
        "    rescale=1./255,\n",
        "    rotation_range=20,\n",
        "    horizontal_flip=True,\n",
        "    validation_split=0.3,\n",
        "    shear_range=0.2,\n",
        "    fill_mode='nearest')"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "batchSize = 4"
      ],
      "metadata": {
        "id": "BeVQwYGLPgvg"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-oMxhmtOUgco",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ca340b30-a4ec-498c-c575-1187d1a5813e"
      },
      "source": [
        "generator_latih = latih_generator.flow_from_directory(\n",
        "    directory=dir_latih,\n",
        "    target_size=(150,150),\n",
        "    batch_size=batchSize,\n",
        "    subset='training',\n",
        "    class_mode='categorical')\n",
        "generator_valid = val_generator.flow_from_directory(\n",
        "    directory=dir_val,\n",
        "    target_size=(150,150),\n",
        "    batch_size=batchSize,\n",
        "    subset='validation',\n",
        "    class_mode='categorical')"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 240 images belonging to 3 classes.\n",
            "Found 9 images belonging to 3 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mC5l4hg4_wv8"
      },
      "source": [
        "model = tensor.keras.models.Sequential([\n",
        "  tensor.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),\n",
        "  tensor.keras.layers.MaxPooling2D(2,2),\n",
        "  tensor.keras.layers.Conv2D(64,(3,3),activation='relu'),\n",
        "  tensor.keras.layers.MaxPooling2D(2,2),\n",
        "  tensor.keras.layers.Conv2D(128,(3,3),activation='relu'),\n",
        "  tensor.keras.layers.MaxPooling2D(2,2),\n",
        "  tensor.keras.layers.Flatten(),\n",
        "  tensor.keras.layers.Dense(512,activation='relu'),\n",
        "  tensor.keras.layers.Dense(3,activation='softmax')\n",
        "  ])"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model_viz = tensor.keras.utils.plot_model(model,\n",
        "                          to_file='model.png',\n",
        "                          show_shapes=True,\n",
        "                          show_layer_names=True,\n",
        "                          rankdir='TB',\n",
        "                          expand_nested=True,\n",
        "                          dpi=55)\n",
        "model_viz"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 630
        },
        "id": "t9BYmfl86p0p",
        "outputId": "78e90a67-74c8-4afd-aba6-6bd5179f6926"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<IPython.core.display.Image object>"
            ],
            "image/png": "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\n"
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rotkCsVQCGcE"
      },
      "source": [
        "model.compile(loss='categorical_crossentropy',optimizer=tensor.optimizers.Adam(),\n",
        "              metrics=['accuracy'])"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "class myCallback(tf.keras.callbacks.Callback):\n",
        "  def on_epoch_end(self, epoch, logs={}):\n",
        "    if(logs.get('accuracy') > 0.95 and logs.get('loss') < 0.1):\n",
        "      print(\"\\nAkurasi di atas 90%, hentikan training!\")\n",
        "      self.model.stop_training = True\n",
        "        \n",
        "stop = myCallback()"
      ],
      "metadata": {
        "id": "Vv_cDRZUw0Uc"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oOrICtyHjzCf"
      },
      "source": [
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "from tensorflow.keras.callbacks import ModelCheckpoint\n",
        "#earstop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)\n",
        "cekmodel = ModelCheckpoint(filepath='model.hdf5',monitor='val_accuracy',mode='max',verbose=1,save_best_only=True)"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BH1w0YmECgvB",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9bc1229b-657b-4492-9024-4c86e663cf37"
      },
      "source": [
        "history = model.fit(\n",
        "    generator_latih,\n",
        "    steps_per_epoch=240/4, # 10243 images = batch_size * steps\n",
        "    epochs=30,\n",
        "    validation_data=generator_valid,\n",
        "    validation_steps=27/4, # 1050 images = batch_size * steps\n",
        "    verbose=2,\n",
        "    callbacks=[cekmodel,stop])"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/30\n",
            "WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 6.75 batches). You may need to use the repeat() function when building your dataset.\n",
            "\n",
            "Epoch 1: val_accuracy improved from -inf to 0.66667, saving model to model.hdf5\n",
            "60/60 - 52s - loss: 1.1586 - accuracy: 0.4083 - val_loss: 1.0735 - val_accuracy: 0.6667 - 52s/epoch - 859ms/step\n",
            "Epoch 2/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.9640 - accuracy: 0.5667 - 37s/epoch - 619ms/step\n",
            "Epoch 3/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.8336 - accuracy: 0.6208 - 37s/epoch - 613ms/step\n",
            "Epoch 4/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.8129 - accuracy: 0.6333 - 37s/epoch - 614ms/step\n",
            "Epoch 5/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.7673 - accuracy: 0.6667 - 37s/epoch - 609ms/step\n",
            "Epoch 6/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.6976 - accuracy: 0.7083 - 37s/epoch - 608ms/step\n",
            "Epoch 7/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.7040 - accuracy: 0.7333 - 36s/epoch - 608ms/step\n",
            "Epoch 8/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.6560 - accuracy: 0.7375 - 36s/epoch - 603ms/step\n",
            "Epoch 9/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.7042 - accuracy: 0.7042 - 36s/epoch - 606ms/step\n",
            "Epoch 10/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.6631 - accuracy: 0.7458 - 36s/epoch - 607ms/step\n",
            "Epoch 11/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.6013 - accuracy: 0.7375 - 36s/epoch - 608ms/step\n",
            "Epoch 12/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.5182 - accuracy: 0.8083 - 36s/epoch - 606ms/step\n",
            "Epoch 13/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.5735 - accuracy: 0.7833 - 36s/epoch - 608ms/step\n",
            "Epoch 14/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.5361 - accuracy: 0.8250 - 36s/epoch - 606ms/step\n",
            "Epoch 15/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.4907 - accuracy: 0.8292 - 37s/epoch - 610ms/step\n",
            "Epoch 16/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.3948 - accuracy: 0.8500 - 36s/epoch - 604ms/step\n",
            "Epoch 17/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.3638 - accuracy: 0.8292 - 37s/epoch - 609ms/step\n",
            "Epoch 18/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.3822 - accuracy: 0.8417 - 36s/epoch - 604ms/step\n",
            "Epoch 19/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.3459 - accuracy: 0.8792 - 36s/epoch - 607ms/step\n",
            "Epoch 20/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.2480 - accuracy: 0.9208 - 37s/epoch - 612ms/step\n",
            "Epoch 21/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.2593 - accuracy: 0.9250 - 36s/epoch - 606ms/step\n",
            "Epoch 22/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.2624 - accuracy: 0.9042 - 37s/epoch - 608ms/step\n",
            "Epoch 23/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.3380 - accuracy: 0.9250 - 36s/epoch - 607ms/step\n",
            "Epoch 24/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.5037 - accuracy: 0.8208 - 37s/epoch - 611ms/step\n",
            "Epoch 25/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.3005 - accuracy: 0.8917 - 36s/epoch - 606ms/step\n",
            "Epoch 26/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.2284 - accuracy: 0.9208 - 36s/epoch - 608ms/step\n",
            "Epoch 27/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 37s - loss: 0.1665 - accuracy: 0.9333 - 37s/epoch - 609ms/step\n",
            "Epoch 28/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.1226 - accuracy: 0.9583 - 36s/epoch - 608ms/step\n",
            "Epoch 29/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.1099 - accuracy: 0.9500 - 36s/epoch - 607ms/step\n",
            "Epoch 30/30\n",
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n",
            "60/60 - 36s - loss: 0.1247 - accuracy: 0.9583 - 36s/epoch - 607ms/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "bxRk07Ieu7C2",
        "outputId": "c2a0794c-d62e-4791-a085-0886b1bbfcaf"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.plot(history.history['accuracy'])\n",
        "plt.plot(history.history['val_accuracy'])\n",
        "plt.title('Akurasi Model Klasifikasi RPS')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.xlabel('Epochs')\n",
        "plt.legend(['train', 'test'], loc='upper left')\n",
        "plt.show()\n",
        "\n",
        "plt.plot(history.history['loss'])\n",
        "plt.plot(history.history['val_loss'])\n",
        "plt.title('Loss Model Klasifikasi RPS')\n",
        "plt.ylabel('Loss')\n",
        "plt.xlabel('Epochs')\n",
        "plt.legend(['train', 'test'], loc='upper left')\n",
        "plt.show()"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TJzX63JavAdT",
        "outputId": "564e46ab-e022-405c-c363-3f1fa44ba1a7"
      },
      "source": [
        "pengubah = tensor.lite.TFLiteConverter.from_keras_model(model)\n",
        "tflite_model = pengubah.convert()"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO:tensorflow:Assets written to: /tmp/tmpg9awh4ki/assets\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:absl:Buffer deduplication procedure will be skipped when flatbuffer library is not properly loaded\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "omHCS6r6vB-i"
      },
      "source": [
        "with tensor.io.gfile.GFile('model.tflite', 'wb') as f:\n",
        "  f.write(tflite_model)"
      ],
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('/content/daun-sayur-buah.hdf5')"
      ],
      "metadata": {
        "id": "JcEDF4H8tbyi"
      },
      "execution_count": 23,
      "outputs": []
    }
  ]
}
