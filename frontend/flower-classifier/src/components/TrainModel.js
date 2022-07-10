import React from 'react';

class TrainModel extends React.Component{
    constructor(props){
        super(props);
        this.state = {
            n_epochs: 1
        }
    }

    callTrainingAPI = () => {

        console.log("N Epochs state: ", this.state.n_epochs);

        let API_BASE_URL = "http://localhost"
        let API_PORT = ":8080";
        let train_route = "/ml_train/";
        let n_epochs = this.state.n_epochs;
        let request_url = API_BASE_URL + API_PORT + train_route + n_epochs;
        
        return new Promise((resolve, reject) => {
            console.log("New Promise. Calling API...");
            return fetch(request_url)
                .then((response) => response.json())
                .then(
                    (data) => {
                        if (data) {
                            console.log("Done! ", data);
                        } else {
                            reject(
                                console.log(
                                    "API request returned empty."
                                )
                            );
                        }
                    },
                    (error) => {
                        reject(new Error("Request failed."));
                    }
                );
        });
    
    }

    handleEpochsChange = (event) => {
        let new_n_epochs = event.target.value 
        if (this.validateEpochsNumber(new_n_epochs)){
            this.setState({n_epochs: new_n_epochs});
            return;
        }
        console.log("Not a valid number. Ignoring...");
    }

    validateEpochsNumber(number){
        // Validate if the text inputed if a number between X and Y
        // Return TRUE if it is, and FALSE otherwise.
        let int_number = parseInt(number)

        return ((int_number >= 1 && int_number <= 10) ? true: false);
    }

    render(){
        return(
            <div className="train-model-container">
                <h2> Train the Model! </h2>
                <p> Select the number of epochs: </p>
                <input type="text"
                    name="epochsValue"
                    onChange={this.handleEpochsChange}
                />

                <br></br>
                <br></br>

                <button onClick={this.callTrainingAPI}>Train!</button>

            </div>
        );
    }
}

export default TrainModel;
