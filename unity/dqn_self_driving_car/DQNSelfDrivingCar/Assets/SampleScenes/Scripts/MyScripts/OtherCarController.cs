using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OtherCarController : MonoBehaviour
{

    public GameObject startPosition;
    public GameObject endPosition;
    private Vector3 origianlPosition;
    private Quaternion originalRotation;
    public float speed = 1.0f;

    // Use this for initialization
    void Start()
    {
        origianlPosition = transform.position;
        originalRotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 forwardDirection = endPosition.GetComponent<Transform>().position - transform.position;
        if (forwardDirection.magnitude < 1)
        {
            MoveToStartPosition();
        }
        else
        {
            forwardDirection = forwardDirection.normalized;
            //transform.Translate(forwardDirection * Time.deltaTime * speed);
            transform.position += forwardDirection * Time.deltaTime * speed;
        }
    }

    public void InitializePosition()
    {
        transform.position = origianlPosition;
        transform.rotation = originalRotation;
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
    }

    void MoveToStartPosition()
    {
        transform.position = startPosition.transform.position;
        transform.rotation = startPosition.transform.rotation;
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
    }
}
