using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Test : MonoBehaviour
{

    public GameObject car;
    bool flag = false;

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Time.time > 50 && flag == false)
        {
            car.GetComponent<UnityStandardAssets.Vehicles.Car.CarController>().Initialize();
            flag = true;
        }

        //Debug.Log(car.GetComponent<Rigidbody>().velocity);
        //var speed = Vector3.Dot(car.GetComponent<Rigidbody>().velocity, car.GetComponent<Transform>().forward);
        //Debug.Log("speed=" + speed);
        //Debug.Log(car.GetComponent<Rigidbody>().angularVelocity);
        //Debug.Log(" ");



        List<float> state = new List<float>();

        //state.Add(Vector3.Dot(car.GetComponent<Rigidbody>().velocity, car.GetComponent<Transform>().forward));
        //state.Add(car.GetComponent<Rigidbody>().angularVelocity.y);

        Vector3 forwardDirection = car.GetComponent<Transform>().forward;
        float forwardDirectionX = forwardDirection.x;
        float forwardDirectionY = forwardDirection.y;
        float forwardDirectionZ = forwardDirection.z;
        int numSensor = 8;

        for (int i = 0; i < numSensor; i++)
        {
            float angle = 2f * Mathf.PI / 8 * i;
            Vector3 sensorDirection = new Vector3(Mathf.Cos(angle) * forwardDirectionX - Mathf.Sin(angle) * forwardDirectionZ,
                                                  forwardDirectionY,
                                                  Mathf.Sin(angle) * forwardDirectionX + Mathf.Cos(angle) * forwardDirectionZ);

            state.Add(MeasureDistance(car.GetComponent<Transform>().position, sensorDirection, 100f));
        }
        for (int i = 0; i < 8; i++)
        {
            Debug.Log(i + ": " + state[i]);
        }


    }
    bool DoIntersect(Vector3 origin, Vector3 normalizedDirection, float a, float b)

    {
        if (Physics.Raycast(origin + normalizedDirection * a, normalizedDirection, b - a) || Physics.Raycast(origin + normalizedDirection * b, -1f * normalizedDirection, b - a))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    float MeasureDistance(Vector3 origin, Vector3 direction, float maxDistance)
    {

        Vector3 normalizedDirection = direction.normalized;
        float distance = 0f;

        //測定範囲内にあるかチェック
        if (DoIntersect(origin, normalizedDirection, 0f, maxDistance) == false)
        {
            distance = maxDistance;
        }
        else
        {
            //二分探索で距離を測定 a <  b
            float b = maxDistance;
            float a = 0f;
            float EPS = 1e-5f;

            while ((b - a) > EPS)
            {
                //2つ以上交点がある場合はよりPSDに近い方を優先する
                if (DoIntersect(origin, normalizedDirection, a, (b + a) / 2) == true)
                {
                    b = (b + a) / 2;
                }
                else
                {
                    a = (b + a) / 2;
                }
            }
            distance = (b + a) / 2;


        }
        return distance;
    }
}
