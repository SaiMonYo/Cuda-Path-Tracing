

class BaseVector{
    float x,y,z;

    BaseVector(float x, float y, float z) : x(x), y(y), z(z) {};
};

class BaseTriangle{
    BaseVector v_0;
    BaseVector v_1;
    BaseVector v_2;

    BaseVector n_0;
    BaseVector n_1;
    BaseVector n_2;

    float u_0, v_0;
    float u_1, v_1;
    float u_2, v_2;
};