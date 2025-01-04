from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

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
    # TODO: Implement your URL checking logic here
    await update.message.reply_text(f"Checking the URL: {url}\nFeature coming soon!")

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
