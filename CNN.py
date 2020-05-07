3
��\Z	  �               @   s�   d Z ddlmZ ddlmZmZmZmZmZm	Z	m
Z
mZmZmZmZmZmZmZmZmZmZmZmZmZmZmZ ddlmZ ddlmZmZmZm Z m!Z!m"Z"m#Z#m$Z$m%Z%m&Z&m'Z'm(Z(m)Z)m*Z*m+Z+m,Z,m-Z-m.Z.m/Z/m0Z0m1Z1m2Z2m3Z3 dS )z@Compatibility namespace for sqlalchemy.sql.schema and related.

�   )�SchemaVisitor)�BLANK_SCHEMA�CheckConstraint�Column�ColumnDefault�
Constraint�DefaultClause�DefaultGenerator�FetchedValue�
ForeignKey�ForeignKeyConstraint�Index�MetaData�PassiveDefault�PrimaryKeyConstraint�
SchemaItem�Sequence�Table�ThreadLocalMetaData�UniqueConstraint�_get_table_key�ColumnCollectionConstraint�ColumnCollectionMixin)�conv)�DDL�CreateTable�	DropTable�CreateSequence�DropSequence�CreateIndex�	DropIndex�CreateSchema�
DropSchema�	_DropView�CreateColumn�AddConstraint�DropConstraint�DDLBase�
DDLElement�_CreateDropBase�_DDLCompiles�sort_tables�sort_tables_and_constraints�SetTableComment�DropTableComment�SetColumnComment�DropColumnCommentN)4�__doc__Zsql.baser   Z
sql.schemar   r   r   r   r   r   r	   r
   r   r   r   r   r   r   r   r   r   r   r   r   r   r   Z
sql.namingr   Zsql.ddlr   r   r   r   r   r   r    r!   r"   r#   r$   r%   r&   r'   r(   r)   r*   r+   r,   r-   r.   r/   r0   � r2   r2   �6D:\env\anaconda\lib\site-packages\sqlalchemy\schema.py�<module>
   s   `                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  