from pydantic import BaseModel


class Query(BaseModel):
    file_path: str
    query: str
    model: str
    username: str


class UserAuth(BaseModel):
    username: str
    password: str
