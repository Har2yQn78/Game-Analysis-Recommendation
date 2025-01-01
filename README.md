# Game Recommendation System

## Project Description
This project is designed to suggest games to Steam users based on their current games. Users can input their owned games, and the system will recommend additional games they might enjoy.

## Installation Instructions

1. **Clone the repository:**
    ```sh
    git clone <https://github.com/Har2yQn78/Game-Recommendation-System.git>
    cd <Game-Recommendation-System>
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
    - Visit [UCSD Steam Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)
    - Download the data and place it in the appropriate directory.

4. **Process the data:**
    - Run the Jupyter notebook for data preprocessing and visualization:
    ```sh
    jupyter notebook Data_preprocess_and_visuilization.ipynb
    ```

5. **Train the model:**
    - This project uses Pytorch to train the recommendation model. Execute the script to train the model base on Collaborative  recommendation system:
    ```sh
    python src\train.py
    ```
   - This project uses Pytorch to train the recommendation model. Execute the script to train the model base on Content-base recommendation system:
       ```sh
       python src\trainCB.py
       ```

## Usage Instructions

1. **Start the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

2. **Interact with the application:**
    - Use the web interface to input your owned games.
    - Get game recommendations based on your input.

## Features
- Suggests games based on user's current game library.
- Allows users to manually input games for personalized recommendations.
- Utilizes a deep learning model built with Pytorch for accurate suggestions.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information
For any questions or issues, please contact [Harry](hamidreza.amiri800@gmail.com).
