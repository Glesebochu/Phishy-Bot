from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from baseline_classifierV2 import model as baseline_model, advanced_model, scaler, extract_additional_features

# Replace 'YOUR_TOKEN' with the bot token from BotFather
TOKEN = '7921896292:AAEzw9EB27bmR0tU2VPc-ozJWpQ2g6X1Jbw'

# Function to predict URL using the chosen model
def predict_url(url, model_choice):
    additional_features = extract_additional_features(url)
    input_data = pd.DataFrame([additional_features])
    
    # Ensure input_data matches the features the scaler and model expect
    expected_features = scaler.feature_names_in_
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Add missing features with default values
    input_data = input_data[expected_features]  # Keep only expected features

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Choose the model
    if model_choice == 'baseline':
        chosen_model = baseline_model
    elif model_choice == 'advanced':
        chosen_model = advanced_model
    else:
        return "Invalid model choice."

    # Make predictions
    predictions = chosen_model.predict(input_data_scaled)
    predicted_probabilities = chosen_model.predict_proba(input_data_scaled)

    result = {
        "Prediction": "Malicious" if predictions[0] == 1 else "Non-Malicious",
        "Prediction Probabilities": predicted_probabilities[0].tolist()
    }
    return result

# Command to handle '/start'
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to Phishy Bot! Use /check <model_choice> <URL> to manually check URLs. Model choices are 'baseline' and 'advanced'."
    )

# Command to handle '/check'
async def check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text(
            "Please provide a model choice and a URL to check. Usage: /check <model_choice> <URL>"
        )
        return

    model_choice = context.args[0].lower()
    url = context.args[1]
    # Use the chosen model to analyze the URL
    result = predict_url(url, model_choice)
    # Communicate the analysis result using the bot
    await update.message.reply_text(f"Analysis result for {url} using {model_choice} model:\n{result}")

def main():
    # Create Application
    application = ApplicationBuilder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("check", check))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
