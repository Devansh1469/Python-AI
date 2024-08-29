from plyer import notification
from win10toast import ToastNotifier
notification.notify(
    title='Learning',
    message='how to show',
    app_name='messages'
)
toaster=ToastNotifier()
toaster.show_toast("title","message",duration=10)
