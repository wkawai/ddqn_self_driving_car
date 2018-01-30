using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DQNAgent : Agent
{
    public GameObject car;
    public GameObject[] targets;
    int targetIndex;
    float prevDistanceToTarget, curDistanceToTarget;
    float curVelocity;
    float curAngularVelocity;
    const int numSensor = 36;
    public GameObject[] otherCars;


    public override List<float> CollectState()
    {
        List<float> state = new List<float>();

        curVelocity = Vector3.Dot(car.GetComponent<Rigidbody>().velocity, car.GetComponent<Transform>().forward);
        curAngularVelocity = car.GetComponent<Rigidbody>().angularVelocity.y;

        state.Add(curVelocity);
        state.Add(curAngularVelocity);


        var directionToTarget = targets[targetIndex].GetComponent<Transform>().position - car.GetComponent<Transform>().position;
        state.Add(Vector3.Dot(directionToTarget.normalized, car.GetComponent<Transform>().forward.normalized));
        state.Add(Vector3.Dot(directionToTarget.normalized, car.GetComponent<Transform>().right.normalized));
        Vector3 backwardDirection = car.GetComponent<Transform>().forward;
        float backwardDirectionX = backwardDirection.x;
        float backwardDirectionY = backwardDirection.y;
        float backwardDirectionZ = backwardDirection.z;

        for (int i = 0; i < numSensor; i++)
        {
            float angle = 2f * Mathf.PI / numSensor * i + Mathf.PI;
            Vector3 sensorDirection = new Vector3(Mathf.Cos(angle) * backwardDirectionX - Mathf.Sin(angle) * backwardDirectionZ,
                                                                      0,
                                                                      Mathf.Sin(angle) * backwardDirectionX + Mathf.Cos(angle) * backwardDirectionZ);

            state.Add(0.01f * MeasureDistance(car.GetComponent<Transform>().position, sensorDirection, 100f));
        }

        return state;
    }

    public override void AgentStep(float[] act)
    {
        float h = 0f, v = 0f;
        if (act[0] == 0)
        {
            //forward
            h = 0f;
            v = 1f;
        }
        else if (act[0] == 1)
        {
            //right
            h = 1f;
            v = 0f;
        }
        else if (act[0] == 2)
        {
            //back
            h = 0f;
            v = -1f;
        }
        else if (act[0] == 3)
        {
            //left
            h = -1f;
            v = 0f;
        }
        else if (act[0] == 4)
        {
            //do nothing
            h = 0f;
            v = 0f;
        }
        else if (act[0] == 5)
        {
            //forward + right
            h = 1f;
            v = 1f;
        }
        else if (act[0] == 6)
        {
            //forward + left
            h = -1f;
            v = 1f;
        }
        else if (act[0] == 7)
        {
            //backward + right
            h = 1f;
            v = -1f;
        }
        else if (act[0] == 8)
        {
            //backward + left
            h = -1f;
            v = -1f;
        }
        car.GetComponent<UnityStandardAssets.Vehicles.Car.CarController>().Move(h, v, v, 0f);

        //reward
        curDistanceToTarget = (car.GetComponent<Transform>().position - targets[targetIndex].GetComponent<Transform>().position).magnitude;
        if (done == false)
        {
            reward = (prevDistanceToTarget - curDistanceToTarget);
        }
        prevDistanceToTarget = curDistanceToTarget;

        //done
        if (car.GetComponent<UnityStandardAssets.Vehicles.Car.CarController>().doCollide == true)
        {
            //Debug.Log(curDistanceToTarget);
            done = true;
            reward = -10f;
            //reward = -50f;
        }

        if (curDistanceToTarget < 20)
        {
            targetIndex++;
            if (targetIndex == 7)
            {
                targetIndex = 0;
            }
            prevDistanceToTarget = curDistanceToTarget = (car.GetComponent<Transform>().position - targets[targetIndex].GetComponent<Transform>().position).magnitude;
        }

    }

    public override void AgentReset()
    {
        car.GetComponent<UnityStandardAssets.Vehicles.Car.CarController>().RandomInitialize(0);
        targetIndex = 0;
        curVelocity = 0f;
        prevDistanceToTarget = curDistanceToTarget = (car.GetComponent<Transform>().position - targets[targetIndex].GetComponent<Transform>().position).magnitude;
        car.GetComponent<UnityStandardAssets.Vehicles.Car.CarController>().doCollide = false;
        foreach(GameObject otherCar in otherCars)
        {
            otherCar.GetComponent<OtherCarController>().InitializePosition();
        }
    }

    public override void AgentOnDone()
    {

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
            float eps = 1e-1f;

            while ((b - a) > eps)
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
