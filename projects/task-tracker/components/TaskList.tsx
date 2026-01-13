'use client';

export interface Task {
  id: string;
  title: string;
  completed: boolean;
}

interface TaskListProps {
  tasks: Task[];
  onToggleComplete: (id: string) => void;
  onDeleteTask: (id: string) => void;
}

export default function TaskList({ tasks, onToggleComplete, onDeleteTask }: TaskListProps) {
  if (tasks.length === 0) {
    return (
      <p className="text-gray-500 dark:text-gray-400 text-center py-8">
        No tasks yet. Add one above!
      </p>
    );
  }

  return (
    <ul className="w-full max-w-md space-y-2">
      {tasks.map((task) => (
        <li
          key={task.id}
          className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
        >
          <input
            type="checkbox"
            checked={task.completed}
            onChange={() => onToggleComplete(task.id)}
            className="w-5 h-5 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
          />
          <span
            className={`flex-1 text-gray-800 dark:text-gray-200 ${
              task.completed ? 'line-through text-gray-400 dark:text-gray-500' : ''
            }`}
          >
            {task.title}
          </span>
          <button
            onClick={() => onDeleteTask(task.id)}
            className="px-3 py-1 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
            aria-label="Delete task"
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
