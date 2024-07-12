#include <QApplication>
#include <QtWebEngineWidgets/QWebEngineView>
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QWebEngineView view;
    view.setUrl(QUrl("http://47.121.186.138:8000/login/"));
    view.show();
    return app.exec();
}
