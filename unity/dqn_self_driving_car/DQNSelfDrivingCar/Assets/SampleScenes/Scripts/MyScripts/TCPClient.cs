using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TCPClient : MonoBehaviour
{

    string sendMsg = "hogehogehoge";
    //サーバーのIPアドレス（または、ホスト名）とポート番号
    string ipOrHost = "127.0.0.1";
    //string ipOrHost = "localhost";
    int port = 10000;

    System.Net.Sockets.TcpClient tcp;

    void Start()
    {
        //TcpClientを作成し、サーバーと接続する
        tcp = new System.Net.Sockets.TcpClient(ipOrHost, port);
        Debug.Log(string.Format("サーバー({0}:{1})と接続しました({2}:{3})。",
            ((System.Net.IPEndPoint)tcp.Client.RemoteEndPoint).Address,
            ((System.Net.IPEndPoint)tcp.Client.RemoteEndPoint).Port,
            ((System.Net.IPEndPoint)tcp.Client.LocalEndPoint).Address,
            ((System.Net.IPEndPoint)tcp.Client.LocalEndPoint).Port));

        //NetworkStreamを取得する
        System.Net.Sockets.NetworkStream ns = tcp.GetStream();

        //読み取り、書き込みのタイムアウトを10秒にする
        //デフォルトはInfiniteで、タイムアウトしない
        //(.NET Framework 2.0以上が必要)
        ns.ReadTimeout = 10000;
        ns.WriteTimeout = 10000;

        //サーバーにデータを送信する
        //文字列をByte型配列に変換
        System.Text.Encoding enc = System.Text.Encoding.UTF8;
        byte[] sendBytes = enc.GetBytes(sendMsg + '\n');
        //データを送信する
        ns.Write(sendBytes, 0, sendBytes.Length);

        //サーバーから送られたデータを受信する
        System.IO.MemoryStream ms = new System.IO.MemoryStream();
        byte[] resBytes = new byte[256];
        int resSize = 0;
        do
        {
            //データの一部を受信する
            resSize = ns.Read(resBytes, 0, resBytes.Length);
            //Readが0を返した時はサーバーが切断したと判断
            if (resSize == 0)
            {
                Debug.Log("サーバーが切断しました。");
                break;
            }
            //受信したデータを蓄積する
            ms.Write(resBytes, 0, resSize);
            //まだ読み取れるデータがあるか、データの最後が\nでない時は、
            // 受信を続ける
        } while (ns.DataAvailable || resBytes[resSize - 1] != '\n');
        //受信したデータを文字列に変換
        string resMsg = enc.GetString(ms.GetBuffer(), 0, (int)ms.Length);
        ms.Close();
        //末尾の\nを削除
        resMsg = resMsg.TrimEnd('\n');
        Debug.Log(resMsg);

        //閉じる
        ns.Close();
        tcp.Close();
        Debug.Log("切断しました。");

        Debug.Log(" ");
    }

    // Update is called once per frame
    void Update()
    {

    }
}
