from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from advanced_classifier import predict_url

# Replace 'YOUR_TOKEN' with the bot token from BotFather
TOKEN = '7921896292:AAEzw9EB27bmR0tU2VPc-ozJWpQ2g6X1Jbw'

# Command to handle '/start'
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to Phishy Bot! Use /check <URL> to manually check URLs."
    )

# Command to handle '/check'
async def check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please provide a URL to check. Usage: /check <URL>"
        )
        return

    url = context.args[0]
    # Use the advanced_model to analyze the URL
    result = predict_url(url)
    # Communicate the analysis result using the bot
    await update.message.reply_text(f"Analysis result for {url}:\n{result}")

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
