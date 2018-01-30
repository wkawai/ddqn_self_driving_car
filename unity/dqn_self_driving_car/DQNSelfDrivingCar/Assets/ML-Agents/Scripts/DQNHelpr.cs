using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DQNHelpr : MonoBehaviour {
    public string fname;
    System.IO.StreamWriter sw;


    private void Awake()
    {
        sw = new System.IO.StreamWriter(fname);
    }
    // Use this for initialization
    void Start () {
        
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void WriteData(string data)
    {
        sw.WriteLine(data);
        sw.Flush();
    }

    private void OnApplicationQuit()
    {
        sw.Close();
    }
}
